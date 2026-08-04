from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from utils.synthetic_lm import (  # noqa: E402
    NUMERIC_FEATURES,
    SyntheticOverlayDataset,
    append_synthetic_edges,
    assert_overlay_preserves_base,
    build_paired_diagnostics,
    build_edge_attr,
    build_scenario_matrix,
    create_matched_controls,
    get_port_role_vector,
    get_protocol_vector,
    operational_availability,
    port_category_index,
    window_id_for_time,
)
from utils.models import SimpleMLP  # noqa: E402


class IdentityScaler:
    def transform(self, values):
        return np.asarray(values)


def _flow_row(**overrides):
    values = {feature: 1.0 for feature in NUMERIC_FEATURES}
    values.update(
        {
            "IN_BYTES": 100.0,
            "OUT_BYTES": 200.0,
            "IN_PKTS": 3.0,
            "OUT_PKTS": 4.0,
            "FLOW_DURATION_MILLISECONDS": 500.0,
            "MIN_IP_PKT_LEN": 40.0,
            "MAX_IP_PKT_LEN": 1500.0,
            "MIN_TTL": 64.0,
            "MAX_TTL": 64.0,
            "FLOW_START_TIME": pd.Timestamp("2018-03-01 14:20:00"),
            "Operational_Available_Time": pd.Timestamp("2018-03-01 14:20:30"),
            "IPV4_SRC_ADDR": "172.31.69.13",
            "IPV4_DST_ADDR": "172.31.69.7",
            "L4_DST_PORT": 445,
            "PROTOCOL": 6,
            "Attack": "Infilteration",
            "Synthetic_Event_ID": "scenario:000",
            "Scenario_ID": "scenario",
            "Synthetic_Stage_ID": 2,
            "Synthetic_Stage": "Lateral Movement",
            "ATTACK_Technique": "T1021.002",
            "Event_Role": "synthetic_lm",
        }
    )
    values.update(overrides)
    return values


def test_protocol_and_port_encodings_match_graph_contract():
    assert get_protocol_vector(6) == [1, 0, 0, 0, 0]
    assert get_protocol_vector(17) == [0, 1, 0, 0, 0]
    assert port_category_index(22) == 1
    assert port_category_index(3389) == 1
    assert port_category_index(445) == 2
    assert port_category_index(49152) == 6
    assert get_port_role_vector(445) == [0, 0, 1, 0, 0, 0, 0]


def test_edge_attr_has_exact_order_and_dimension():
    frame = pd.DataFrame([_flow_row()])
    edge_attr = build_edge_attr(frame, IdentityScaler())
    assert edge_attr.shape == (1, 32)
    assert edge_attr[0, :7].tolist() == [0, 0, 1, 0, 0, 0, 0]
    assert edge_attr[0, 7:12].tolist() == [1, 0, 0, 0, 0]
    expected_numeric = np.log1p(frame.loc[:, NUMERIC_FEATURES].to_numpy(dtype=float))
    np.testing.assert_allclose(edge_attr[0, 12:].numpy(), expected_numeric[0], rtol=1e-6)


def test_scenario_matrix_contains_54_isolated_scenarios():
    rows = []
    for protocol in ("SMB_RPC", "RDP", "SSH"):
        for rank in (1, 2, 3):
            rows.append(
                {
                    "Protocol": protocol,
                    "Target_Rank": rank,
                    "Target_IP": f"172.31.{rank}.{10 + rank}",
                    "Discovery_Time": pd.Timestamp("2018-03-01 14:10:00"),
                }
            )
    scenarios = build_scenario_matrix(
        pd.DataFrame(rows),
        global_start=pd.Timestamp("2018-03-01 00:00:00"),
        global_end=pd.Timestamp("2018-03-01 23:30:00"),
    )
    assert len(scenarios) == 54
    assert scenarios["scenario_id"].is_unique
    assert set(scenarios["horizon_minutes"]) == {5, 15, 30}
    assert set(scenarios["access_path"]) == {"valid_account", "authentication_attempts"}


def test_operational_availability_uses_later_of_window_and_flow_end():
    global_start = pd.Timestamp("2018-03-01 00:00:03.289")
    start = pd.Timestamp("2018-03-01 14:20:10")
    window_limited = operational_availability(start, 500, global_start)
    assert window_limited == pd.Timestamp("2018-03-01 14:20:33.289")
    flow_limited = operational_availability(start, 60_000, global_start)
    assert flow_limited == pd.Timestamp("2018-03-01 14:21:10")
    assert window_id_for_time(global_start, global_start) == 0


def test_append_only_overlay_preserves_original_graph_tensors():
    base = Data(
        x=torch.ones((2, 16), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.zeros((1, 32), dtype=torch.float32),
        y=torch.tensor([0.0]),
    )
    base.global_node_ids = torch.tensor([10, 11], dtype=torch.long)
    base.timestamp = pd.Timestamp("2018-03-01 14:20:03.289")
    base.is_empty = False
    flows = pd.DataFrame([_flow_row()])
    overlay = append_synthetic_edges(
        base,
        flows,
        ip_to_id={"172.31.69.13": 10, "172.31.69.7": 12},
        scaler=IdentityScaler(),
    )
    assert_overlay_preserves_base(base, overlay)
    assert overlay.edge_index.shape[1] == 2
    assert overlay.x.shape[0] == 3
    assert overlay.synthetic_mask.tolist() == [False, True]
    assert overlay.synthetic_roles[-1] == "synthetic_lm"


def test_sparse_overlay_dataset_falls_back_to_base(tmp_path):
    base_dir = tmp_path / "base" / "test2"
    overlay_dir = tmp_path / "overlay" / "test2"
    base_dir.mkdir(parents=True)
    overlay_dir.mkdir(parents=True)

    for window in range(2):
        graph = Data(
            x=torch.ones((1, 16)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 32)),
            y=torch.empty((0,)),
        )
        graph.global_node_ids = torch.tensor([window])
        graph.timestamp = pd.Timestamp("2018-03-01") + pd.Timedelta(seconds=30 * window)
        torch.save(graph, base_dir / f"graph_{window:06d}.pt")

    changed = torch.load(base_dir / "graph_000001.pt", weights_only=False)
    changed.global_node_ids = torch.tensor([999])
    torch.save(changed, overlay_dir / "graph_000001.pt")

    dataset = SyntheticOverlayDataset(tmp_path / "base", tmp_path / "overlay")
    assert dataset[0].global_node_ids.tolist() == [0]
    assert dataset[1].global_node_ids.tolist() == [999]


def test_mlp_is_invariant_to_endpoint_permutation():
    torch.manual_seed(7)
    model = SimpleMLP(node_dim=16, edge_dim=32, hidden_dim=32, dropout=0.2)
    model.eval()
    edge_attr = torch.randn((3, 32))
    x_a = torch.ones((3, 16))
    x_b = torch.ones((5, 16))
    edge_index_a = torch.tensor([[0, 0, 1], [1, 2, 2]])
    edge_index_b = torch.tensor([[4, 3, 2], [0, 1, 0]])
    with torch.no_grad():
        output_a = model(x_a, edge_index_a, edge_attr)
        output_b = model(x_b, edge_index_b, edge_attr)
    torch.testing.assert_close(output_a, output_b)


def test_controls_cover_every_attack_with_multiple_sources():
    scenarios = pd.DataFrame(
        [
            {
                "scenario_id": "attack_a",
                "scenario_type": "attack",
                "protocol": "SSH",
                "pivot_ip": "172.31.69.13",
                "target_ip": "172.31.69.7",
                "horizon_minutes": 5,
                "access_path": "valid_account",
            },
            {
                "scenario_id": "attack_b",
                "scenario_type": "attack",
                "protocol": "RDP",
                "pivot_ip": "172.31.69.13",
                "target_ip": "172.31.69.14",
                "horizon_minutes": 30,
                "access_path": "authentication_attempts",
            },
        ]
    )
    attack_flows = pd.DataFrame(
        [
            _flow_row(
                Scenario_ID="attack_a",
                Synthetic_Event_ID="attack_a:000",
                Linked_Attack_Event_ID="attack_a:000",
                Scenario_Type="attack",
                Protocol_Mechanism="SSH",
                Access_Path="valid_account",
                Horizon_Minutes=5,
                Target_IP="172.31.69.7",
            ),
            _flow_row(
                Scenario_ID="attack_b",
                Synthetic_Event_ID="attack_b:000",
                Linked_Attack_Event_ID="attack_b:000",
                Scenario_Type="attack",
                Protocol_Mechanism="RDP",
                Access_Path="authentication_attempts",
                Horizon_Minutes=30,
                Target_IP="172.31.69.14",
                IPV4_DST_ADDR="172.31.69.14",
            ),
        ]
    )
    relevant = pd.DataFrame(
        {
            "IPV4_SRC_ADDR": ["172.31.10.1", "172.31.10.2", "172.31.10.3"],
            "IPV4_DST_ADDR": ["172.31.20.1", "172.31.20.1", "172.31.20.1"],
            "_src_internal": [True, True, True],
            "_dst_internal": [True, True, True],
            "Attack": ["Infilteration", "Benign", "Benign"],
        }
    )
    ip_to_id = {
        "172.31.10.1": 1,
        "172.31.10.2": 2,
        "172.31.10.3": 3,
        "172.31.20.1": 4,
    }
    controls, manifest = create_matched_controls(
        attack_flows,
        scenarios,
        relevant,
        ip_to_id,
        controls_per_attack=2,
    )
    assert len(manifest) == 4
    assert manifest.groupby("linked_attack_scenario").size().eq(2).all()
    assert len(controls) == 4
    assert set(controls["Linked_Attack_Event_ID"]) == {"attack_a:000", "attack_b:000"}
    assert controls["Attack"].eq("Benign").all()
    assert not controls["IPV4_SRC_ADDR"].eq("172.31.69.13").any()
    assert not controls["IPV4_SRC_ADDR"].eq("172.31.10.1").any()


def test_paired_diagnostics_match_identical_event_provenance():
    attack = _flow_row(
        Scenario_ID="attack_a",
        Synthetic_Event_ID="attack_a:000",
        Linked_Attack_Event_ID="attack_a:000",
        Scenario_Type="attack",
        Protocol_Mechanism="SSH",
        Access_Path="valid_account",
        Horizon_Minutes=15,
        Target_IP="172.31.69.7",
        Donor_Row_ID=123,
        Donor_Original_Attack="Benign",
    )
    controls = []
    for replicate, prediction in ((1, 0), (2, 1)):
        row = dict(attack)
        row.update(
            {
                "Scenario_ID": f"control_r{replicate:02d}_attack_a",
                "Synthetic_Event_ID": f"control_r{replicate:02d}_attack_a:000",
                "Scenario_Type": "control",
                "Linked_Attack_Scenario": "attack_a",
                "Control_Replicate": replicate,
                "Control_Source_IP": f"172.31.10.{replicate}",
                "_prediction": prediction,
            }
        )
        controls.append(row)
    flows = pd.DataFrame([attack, *controls]).drop(columns="_prediction")
    prediction_rows = [
        {
            "Model": "ST-GNN",
            "Synthetic_Event_ID": "attack_a:000",
            "Scenario_ID": "attack_a",
            "Is_Synthetic": True,
            "Synthetic_Role": "synthetic_lm",
            "Probability": 0.8,
            "y_pred": 1,
            "Source_IP": "172.31.69.13",
            "Diagnostic_Ablation": "full",
        }
    ]
    for replicate, prediction in ((1, 0), (2, 1)):
        prediction_rows.append(
            {
                "Model": "ST-GNN",
                "Synthetic_Event_ID": f"control_r{replicate:02d}_attack_a:000",
                "Scenario_ID": f"control_r{replicate:02d}_attack_a",
                "Is_Synthetic": True,
                "Synthetic_Role": "synthetic_lm",
                "Probability": 0.2 if prediction == 0 else 0.7,
                "y_pred": prediction,
                "Source_IP": f"172.31.10.{replicate}",
                "Diagnostic_Ablation": "full",
            }
        )
    paired = build_paired_diagnostics(pd.DataFrame(prediction_rows), flows)
    assert len(paired) == 2
    assert set(paired["Paired_Outcome"]) == {"linked_only", "both_positive"}
    assert paired["Linked_Probability"].eq(0.8).all()
    assert set(paired["Control_Probability"]) == {0.2, 0.7}
