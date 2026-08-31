#!/usr/bin/env python3
"""Run resumable validation-only Phase-4B calibration screening in Colab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = SCRIPT_DIR.parent
REPOSITORY_ROOT = PYTHON_ROOT.parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from utils.calibration import git_code_revision, sha256_file  # noqa: E402
from utils.calibration_screening import (  # noqa: E402
    DEFAULT_AP_EQUIVALENCE_MARGIN,
    SCREENING_PROFILE,
    SCREENING_SEED,
    build_screening_model_config,
    build_screening_plan,
    build_screening_summary,
    load_calibration_manifest,
    result_from_run_record,
    select_stage_candidates,
    validate_calibration_sources,
    write_json_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("mlp", "stgnn"), required=True)
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="ST-GNN requires the complete explicit shortlist. MLP always runs all candidates.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--node-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-steps", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--ap-equivalence-margin",
        type=float,
        default=DEFAULT_AP_EQUIVALENCE_MARGIN,
        help="Frozen absolute validation-AP margin used to build the shortlist.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate inputs and freeze the plan without starting training.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Deliberately rerun candidates that already have completion records.",
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
    stage: str,
    candidate_id: str,
    calibration_manifest_sha256: str,
    expected_configuration: Mapping[str, Any],
) -> bool:
    try:
        configuration = record["configuration"]
        identity_matches = (
            configuration["data_params"]["calibration_candidate_id"] == candidate_id
            and configuration["data_params"]["calibration_manifest_sha256"]
            == calibration_manifest_sha256
            and configuration["extra_params"]["screening_stage"] == stage
            and int(configuration["extra_params"]["seed"]) == SCREENING_SEED
            and configuration["selection_metric"] == "average_precision"
            and configuration["threshold"] == {"strategy": "max_f1"}
        )
        if not identity_matches:
            return False
        for key in (
            "type",
            "variant",
            "model_params",
            "temporal",
            "temporal_memory_policy",
            "selection_metric",
            "threshold",
        ):
            if configuration.get(key) != expected_configuration.get(key):
                return False
        if any(
            configuration["data_params"].get(key) != value
            for key, value in expected_configuration["data_params"].items()
        ):
            return False
        if any(
            configuration["extra_params"].get(key) != value
            for key, value in expected_configuration["extra_params"].items()
        ):
            return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _find_latest_completed_run(
    candidate_root: Path,
    *,
    stage: str,
    candidate_id: str,
    calibration_manifest_sha256: str,
    expected_configuration: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], Path] | None:
    matches: list[tuple[Path, dict[str, Any], Path]] = []
    for record_path in (candidate_root / "run_records").glob("*.json"):
        record = _load_json(record_path)
        if not _record_matches(
            record,
            stage=stage,
            candidate_id=candidate_id,
            calibration_manifest_sha256=calibration_manifest_sha256,
            expected_configuration=expected_configuration,
        ):
            continue
        checkpoints = sorted((candidate_root / "models").glob(f"{record['run_id']}_*.pth"))
        if len(checkpoints) != 1:
            continue
        matches.append((record_path, record, checkpoints[0]))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0].stat().st_mtime_ns)


def _completion_from_files(
    *,
    record_path: Path,
    record: Mapping[str, Any],
    checkpoint_path: Path,
    candidate: Mapping[str, Any],
    stage: str,
    calibration_manifest_sha256: str,
    screening_plan_sha256: str,
    results_root: Path,
) -> dict[str, Any]:
    return result_from_run_record(
        record,
        candidate=candidate,
        stage=stage,
        calibration_manifest_sha256=calibration_manifest_sha256,
        screening_plan_sha256=screening_plan_sha256,
        run_record_path=record_path.relative_to(results_root).as_posix(),
        run_record_sha256=sha256_file(record_path),
        checkpoint_path=checkpoint_path.relative_to(results_root).as_posix(),
        checkpoint_sha256=sha256_file(checkpoint_path),
    )


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")

    import torch
    from torch_geometric.loader import DataLoader

    from utils.datasets import NF_IDS_Dataset
    from utils.experiment import ExperimentManager
    from utils.models import ST_GNN_Identity, SimpleMLP
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
    candidates = select_stage_candidates(
        calibration, args.stage, tuple(args.candidate_id)
    )

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

    configurations = [
        build_screening_model_config(
            stage=args.stage,
            candidate=candidate,
            edge_dim=train_dataset.edge_dim,
            window_ms=train_dataset.window_ms,
            correction_rule_version=calibration["source_artifacts"][
                "correction_rule_version"
            ],
            calibration_manifest_sha256=calibration_sha256,
            calibration_code_revision=calibration["code_revision"],
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            node_dim=args.node_dim,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            batch_steps=args.batch_steps,
            patience=args.patience,
            min_delta=args.min_delta,
        )
        for candidate in candidates
    ]
    code_revision = git_code_revision(REPOSITORY_ROOT)
    plan = build_screening_plan(
        stage=args.stage,
        calibration_manifest_sha256=calibration_sha256,
        graph_manifest_sha256=graph_manifest_sha256,
        code_revision=code_revision,
        configurations=configurations,
        device=str(device),
        num_workers=args.num_workers,
        ap_equivalence_margin=args.ap_equivalence_margin,
    )
    stage_root = results_root / args.stage
    plan_path = stage_root / "screening_plan.json"
    plan_status = write_json_artifact(plan, plan_path)
    plan_sha256 = sha256_file(plan_path)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "screening_plan": str(plan_path),
                "screening_plan_status": plan_status,
                "screening_plan_sha256": plan_sha256,
                "candidates": [candidate["candidate_id"] for candidate in candidates],
                "device": str(device),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.plan_only:
        print("Plan-only validation completed; no model was trained.")
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
    model_class = SimpleMLP if args.stage == "mlp" else ST_GNN_Identity

    for candidate, configuration in zip(candidates, configurations):
        candidate_id = candidate["candidate_id"]
        candidate_root = stage_root / candidate_id
        completion_path = candidate_root / "completion.json"
        existing = _find_latest_completed_run(
            candidate_root,
            stage=args.stage,
            candidate_id=candidate_id,
            calibration_manifest_sha256=calibration_sha256,
            expected_configuration=configuration,
        )
        if completion_path.is_file() and existing is not None and not args.rerun:
            record_path, record, checkpoint_path = existing
            recovered_completion = _completion_from_files(
                record_path=record_path,
                record=record,
                checkpoint_path=checkpoint_path,
                candidate=candidate,
                stage=args.stage,
                calibration_manifest_sha256=calibration_sha256,
                screening_plan_sha256=plan_sha256,
                results_root=results_root,
            )
            write_json_artifact(recovered_completion, completion_path)
            print(f"Skipping completed candidate {candidate_id}.")
            continue
        if existing is None or args.rerun:
            experiment_name = f"phase4b_{args.stage}_{candidate_id}"
            manager = ExperimentManager(
                log_file=candidate_root / "logs" / "run_metrics.csv",
                model_dir=candidate_root / "models",
                record_dir=candidate_root / "run_records",
            )
            run_multiple_seeds(
                model_class=model_class,
                model_config=configuration,
                train_loader=train_loader,
                val_loader=val_loader,
                manager=manager,
                seeds=(SCREENING_SEED,),
                epochs=args.epochs,
                device=device,
                experiment_name=experiment_name,
                json_dir=candidate_root / "histories",
                plots_dir=candidate_root / "plots",
            )
            existing = _find_latest_completed_run(
                candidate_root,
                stage=args.stage,
                candidate_id=candidate_id,
                calibration_manifest_sha256=calibration_sha256,
                expected_configuration=configuration,
            )
        if existing is None:
            raise RuntimeError(f"No completed run record/checkpoint found for {candidate_id}.")
        record_path, record, checkpoint_path = existing
        completion = _completion_from_files(
            record_path=record_path,
            record=record,
            checkpoint_path=checkpoint_path,
            candidate=candidate,
            stage=args.stage,
            calibration_manifest_sha256=calibration_sha256,
            screening_plan_sha256=plan_sha256,
            results_root=results_root,
        )
        write_json_artifact(completion, completion_path, overwrite=args.rerun)
        print(
            f"Completed {candidate_id}: validation AP={completion['best_validation_ap']:.6f}, "
            f"threshold={completion['selected_validation_threshold']:.6f}"
        )

    completed_results = [
        _load_json(path)
        for path in sorted(stage_root.glob("calibration_*/completion.json"))
    ]
    final_summary = build_screening_summary(
        stage=args.stage,
        calibration_manifest_sha256=calibration_sha256,
        plan_sha256=plan_sha256,
        expected_candidate_ids=[item["candidate_id"] for item in candidates],
        results=completed_results,
        ap_equivalence_margin=args.ap_equivalence_margin,
    )
    write_json_artifact(
        final_summary, stage_root / "screening_summary.json", overwrite=True
    )
    print(json.dumps(final_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
