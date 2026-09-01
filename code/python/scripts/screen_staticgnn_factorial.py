#!/usr/bin/env python3
"""Run the resumable 20-cell StaticGNN calibration factorial on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = SCRIPT_DIR.parent
REPOSITORY_ROOT = PYTHON_ROOT.parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from utils.calibration import git_code_revision, sha256_file  # noqa: E402
from utils.calibration_screening import (  # noqa: E402
    DEFAULT_AP_EQUIVALENCE_MARGIN,
    SCREENING_PROFILE,
    calibration_candidates,
    load_calibration_manifest,
    validate_calibration_sources,
    write_json_artifact,
)
from utils.staticgnn_factorial import (  # noqa: E402
    DEFAULT_FINALIST_COUNT,
    DEFAULT_HIDDEN_DIMS,
    DEFAULT_LEARNING_RATES,
    FACTORIAL_SEED,
    FACTORIAL_STAGE,
    build_staticgnn_factorial_configurations,
    build_staticgnn_factorial_plan,
    build_staticgnn_factorial_summary,
    factorial_result_from_run_record,
    factorial_run_configuration_matches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--node-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--learning-rate",
        type=float,
        action="append",
        default=[],
        help="Repeat to define the factorial learning-rate axis.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        action="append",
        default=[],
        help="Repeat to define the factorial hidden-width axis.",
    )
    parser.add_argument("--batch-steps", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--ap-equivalence-margin",
        type=float,
        default=DEFAULT_AP_EQUIVALENCE_MARGIN,
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Deliberately rerun configurations that already have verified artifacts.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _record_matches(
    record: Mapping[str, Any],
    *,
    expected_configuration: Mapping[str, Any],
) -> bool:
    """Match the immutable factorial fields while allowing run metadata additions."""
    try:
        actual = record["configuration"]
    except (KeyError, TypeError):
        return False
    return factorial_run_configuration_matches(actual, expected_configuration)


def _find_latest_completed_run(
    configuration_root: Path,
    *,
    expected_configuration: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], Path] | None:
    matches: list[tuple[Path, dict[str, Any], Path]] = []
    for record_path in (configuration_root / "run_records").glob("*.json"):
        record = _load_json(record_path)
        if not _record_matches(record, expected_configuration=expected_configuration):
            continue
        checkpoints = sorted(
            (configuration_root / "models").glob(f"{record['run_id']}_*.pth")
        )
        if len(checkpoints) == 1:
            matches.append((record_path, record, checkpoints[0]))
    return max(matches, key=lambda item: item[0].stat().st_mtime_ns) if matches else None


def _completion_from_files(
    *,
    record_path: Path,
    record: Mapping[str, Any],
    checkpoint_path: Path,
    expected_configuration: Mapping[str, Any],
    calibration_manifest_sha256: str,
    plan_sha256: str,
    results_root: Path,
) -> dict[str, Any]:
    return factorial_result_from_run_record(
        record,
        expected_configuration=expected_configuration,
        calibration_manifest_sha256=calibration_manifest_sha256,
        factorial_plan_sha256=plan_sha256,
        run_record_path=record_path.relative_to(results_root).as_posix(),
        run_record_sha256=sha256_file(record_path),
        checkpoint_path=checkpoint_path.relative_to(results_root).as_posix(),
        checkpoint_sha256=sha256_file(checkpoint_path),
    )


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")

    import torch
    from torch_geometric.loader import DataLoader

    from utils.datasets import NF_IDS_Dataset
    from utils.experiment import ExperimentManager
    from utils.models import StaticGNN_Identity
    from utils.training import run_multiple_seeds

    graph_root = args.graph_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    calibration, calibration_sha256 = load_calibration_manifest(
        args.calibration_manifest
    )
    graph_manifest_path = graph_root / "graph_manifest.json"
    graph_manifest = _load_json(graph_manifest_path)
    graph_manifest_sha256 = sha256_file(graph_manifest_path)
    validate_calibration_sources(
        calibration,
        graph_manifest,
        actual_graph_manifest_sha256=graph_manifest_sha256,
    )

    learning_rates = tuple(args.learning_rate or DEFAULT_LEARNING_RATES)
    hidden_dims = tuple(args.hidden_dim or DEFAULT_HIDDEN_DIMS)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    train_dataset = NF_IDS_Dataset(
        graph_root, SCREENING_PROFILE, "train", verify_checksums=args.verify_checksums
    )
    val_dataset = NF_IDS_Dataset(
        graph_root, SCREENING_PROFILE, "val", verify_checksums=args.verify_checksums
    )
    if train_dataset.edge_dim != val_dataset.edge_dim:
        raise ValueError("Train and validation feature dimensions differ.")

    candidates = list(calibration_candidates(calibration).values())
    configurations = build_staticgnn_factorial_configurations(
        candidates=candidates,
        learning_rates=learning_rates,
        hidden_dims=hidden_dims,
        edge_dim=train_dataset.edge_dim,
        node_dim=args.node_dim,
        window_ms=train_dataset.window_ms,
        correction_rule_version=calibration["source_artifacts"][
            "correction_rule_version"
        ],
        calibration_manifest_sha256=calibration_sha256,
        calibration_code_revision=calibration["code_revision"],
        epochs=args.epochs,
        dropout=args.dropout,
        batch_steps=args.batch_steps,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    plan = build_staticgnn_factorial_plan(
        calibration_manifest_sha256=calibration_sha256,
        graph_manifest_sha256=graph_manifest_sha256,
        code_revision=git_code_revision(REPOSITORY_ROOT),
        configurations=configurations,
        learning_rates=learning_rates,
        hidden_dims=hidden_dims,
        device=str(device),
        num_workers=args.num_workers,
        finalist_count=DEFAULT_FINALIST_COUNT,
        ap_equivalence_margin=args.ap_equivalence_margin,
    )
    stage_root = results_root / FACTORIAL_STAGE
    plan_path = stage_root / "factorial_plan.json"
    plan_status = write_json_artifact(plan, plan_path)
    plan_sha256 = sha256_file(plan_path)
    print(
        json.dumps(
            {
                "stage": FACTORIAL_STAGE,
                "factorial_plan": str(plan_path),
                "factorial_plan_status": plan_status,
                "factorial_plan_sha256": plan_sha256,
                "configuration_count": len(configurations),
                "learning_rates": learning_rates,
                "hidden_dims": hidden_dims,
                "seed": FACTORIAL_SEED,
                "device": str(device),
                "test_splits_accessed": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.plan_only:
        print("Plan-only validation completed; no model was trained.", flush=True)
        return

    persistent_workers = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=persistent_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=persistent_workers,
        pin_memory=False,
    )

    invocation_started = time.perf_counter()
    executed_durations: list[float] = []
    total = len(configurations)
    for index, configuration in enumerate(configurations, start=1):
        data_params = configuration["data_params"]
        model_params = configuration["model_params"]
        extra_params = configuration["extra_params"]
        configuration_id = data_params["factorial_configuration_id"]
        configuration_root = stage_root / configuration_id
        completion_path = configuration_root / "completion.json"
        existing = _find_latest_completed_run(
            configuration_root,
            expected_configuration=configuration,
        )
        if completion_path.is_file() and existing is None and not args.rerun:
            raise RuntimeError(
                f"{completion_path} exists, but its run record/checkpoint cannot be "
                "verified. Preserve the artifacts and inspect the inconsistency; "
                "normal resume will not overwrite it."
            )
        if completion_path.is_file() and existing is not None and not args.rerun:
            record_path, record, checkpoint_path = existing
            completion = _completion_from_files(
                record_path=record_path,
                record=record,
                checkpoint_path=checkpoint_path,
                expected_configuration=configuration,
                calibration_manifest_sha256=calibration_sha256,
                plan_sha256=plan_sha256,
                results_root=results_root,
            )
            write_json_artifact(completion, completion_path)
            print(
                f"[{index:02d}/{total:02d}] SKIP {configuration_id} "
                "(verified completion record and checkpoint)",
                flush=True,
            )
            continue

        print("\n" + "=" * 88, flush=True)
        print(
            f"[{index:02d}/{total:02d}] START {configuration_id} "
            f"| pos_weight={extra_params['pos_weight']:.6f} "
            f"| bias={model_params['output_bias_init']:.6f} "
            f"| lr={extra_params['learning_rate']:.6g} "
            f"| hidden_dim={model_params['hidden_dim']} "
            f"| seed={FACTORIAL_SEED} | device={device}",
            flush=True,
        )
        print("* marks a new best validation-AP checkpoint.", flush=True)
        print("=" * 88, flush=True)
        configuration_started = time.perf_counter()
        if existing is None or args.rerun:
            manager = ExperimentManager(
                log_file=configuration_root / "logs" / "run_metrics.csv",
                model_dir=configuration_root / "models",
                record_dir=configuration_root / "run_records",
            )
            run_multiple_seeds(
                model_class=StaticGNN_Identity,
                model_config=configuration,
                train_loader=train_loader,
                val_loader=val_loader,
                manager=manager,
                seeds=(FACTORIAL_SEED,),
                epochs=args.epochs,
                device=device,
                experiment_name=f"phase4b_{configuration_id}",
                json_dir=configuration_root / "histories",
                plots_dir=configuration_root / "plots",
            )
            existing = _find_latest_completed_run(
                configuration_root,
                expected_configuration=configuration,
            )
        if existing is None:
            raise RuntimeError(
                f"No completed run record/checkpoint found for {configuration_id}."
            )
        record_path, record, checkpoint_path = existing
        completion = _completion_from_files(
            record_path=record_path,
            record=record,
            checkpoint_path=checkpoint_path,
            expected_configuration=configuration,
            calibration_manifest_sha256=calibration_sha256,
            plan_sha256=plan_sha256,
            results_root=results_root,
        )
        write_json_artifact(completion, completion_path, overwrite=args.rerun)
        executed_durations.append(time.perf_counter() - configuration_started)
        remaining = total - index
        eta = (
            sum(executed_durations) / len(executed_durations) * remaining
            if executed_durations
            else 0.0
        )
        print(
            f"[{index:02d}/{total:02d}] COMPLETE {configuration_id} "
            f"| best val AP={completion['best_validation_ap']:.6f} "
            f"| threshold={completion['selected_validation_threshold']:.6f} "
            f"({completion['threshold_selection']}) "
            f"| run time={_format_duration(completion['timing_seconds']['total'])}",
            flush=True,
        )
        print(
            f"Artifacts | completion={completion_path} "
            f"| record={record_path} | checkpoint={checkpoint_path}",
            flush=True,
        )
        if remaining:
            print(
                f"Progress | {index}/{total} processed this pass "
                f"| elapsed={_format_duration(time.perf_counter() - invocation_started)} "
                f"| approximate remaining={_format_duration(eta)}",
                flush=True,
            )

    completed_results = [
        _load_json(path)
        for path in sorted(stage_root.glob("*/completion.json"))
    ]
    expected_ids = [
        config["data_params"]["factorial_configuration_id"]
        for config in configurations
    ]
    summary = build_staticgnn_factorial_summary(
        calibration_manifest_sha256=calibration_sha256,
        plan_sha256=plan_sha256,
        expected_configuration_ids=expected_ids,
        results=completed_results,
        finalist_count=DEFAULT_FINALIST_COUNT,
    )
    summary_path = stage_root / "factorial_summary.json"
    write_json_artifact(summary, summary_path, overwrite=True)
    print("\n" + "=" * 88, flush=True)
    print("STATICGNN FACTORIAL SUMMARY", flush=True)
    for rank, result in enumerate(summary["ranking"], start=1):
        print(
            f"{rank:02d}. {result['configuration_id']} "
            f"| AP={result['best_validation_ap']:.6f} "
            f"| weight={result['pos_weight']:.6f} "
            f"| lr={result['learning_rate']:.6g} "
            f"| hidden={result['hidden_dim']} "
            f"| time={_format_duration(result['timing_seconds']['total'])}",
            flush=True,
        )
    print(
        "Recommended finalists: "
        f"{summary['recommended_finalist_configuration_ids']}",
        flush=True,
    )
    print(f"Summary saved: {summary_path}", flush=True)
    print(
        f"Invocation elapsed: {_format_duration(time.perf_counter() - invocation_started)}",
        flush=True,
    )
    print("=" * 88, flush=True)


if __name__ == "__main__":
    main()
