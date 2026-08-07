import os
import sys
import unittest

import pandas as pd
import torch


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLEXIBLE_DIR = os.path.join(ROOT_DIR, "flexible model")
for path in (ROOT_DIR, FLEXIBLE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmarking_common.hgt_depth import (  # noqa: E402
    CELL_SIMILARITY,
    DRUG_RESPONSE,
    DRUG_SIMILARITY,
    annotate_pair_distances,
    assert_response_edges_exclude_pairs,
)
from benchmarking_common.splits import (  # noqa: E402
    create_fusecdr_folds,
    create_historical_random_folds,
)
from model_flexible import EncoderConfig, FUSECDR  # noqa: E402


METADATA = (
    ["drug", "cell"],
    [DRUG_RESPONSE, CELL_SIMILARITY, DRUG_SIMILARITY],
)


class HistoricalRandomSplitTests(unittest.TestCase):
    def setUp(self):
        self.pairs = pd.DataFrame(
            {
                "cell_id": [f"C{index:02d}" for index in range(25)],
                "drug_id": [f"D{index % 5:02d}" for index in range(25)],
                "label": [index % 2 for index in range(25)],
            }
        )

    def test_historical_split_is_deterministic_and_complete(self):
        first = create_historical_random_folds(self.pairs, seed=0, n_splits=5)
        second = create_historical_random_folds(self.pairs, seed=0, n_splits=5)
        expected_pairs = set(map(tuple, self.pairs[["cell_id", "drug_id", "label"]].to_numpy()))
        for first_fold, second_fold in zip(first, second):
            observed = []
            for split_name in ("train", "val", "test"):
                pd.testing.assert_frame_equal(
                    first_fold[split_name],
                    second_fold[split_name],
                )
                observed.append(
                    set(map(tuple, first_fold[split_name][["cell_id", "drug_id", "label"]].to_numpy()))
                )
            self.assertFalse(observed[0] & observed[1])
            self.assertFalse(observed[0] & observed[2])
            self.assertFalse(observed[1] & observed[2])
            self.assertEqual(set().union(*observed), expected_pairs)

    def test_historical_validation_draw_differs_from_newer_policy(self):
        historical = create_historical_random_folds(self.pairs, seed=0, n_splits=5)
        newer = create_fusecdr_folds(self.pairs, seed=0, n_splits=5)
        self.assertTrue(
            any(
                not historical[index]["val"].equals(newer[index]["val"])
                for index in range(5)
            )
        )


class HGTDepthConfigurationTests(unittest.TestCase):
    def _build_model(self, **depth_args):
        return FUSECDR(
            atom_shape=5,
            encoder_configs=[
                EncoderConfig(
                    category="genomics",
                    subtype="mutation",
                    encoder_type="genomics_fc",
                    input_dim=4,
                    output_dim=8,
                )
            ],
            metadata=METADATA,
            hidden_dim=8,
            output_dim=4,
            fusion_dim=8,
            heads=2,
            drug_encoder_type="fingerprint",
            drug_input_dim=4,
            **depth_args,
        )

    def test_legacy_num_layers_controls_both_branches(self):
        model = self._build_model(num_layers=2)
        self.assertEqual(len(model.local_convs), 2)
        self.assertEqual(len(model.global_convs), 2)
        self.assertTrue(model.use_local_branch)
        self.assertTrue(model.use_global_branch)
        self.assertIsNotNone(model.fusion)

    def test_explicit_two_branch_depths_preserve_legacy_full_model(self):
        torch.manual_seed(17)
        legacy = self._build_model(num_layers=2)
        torch.manual_seed(17)
        explicit = self._build_model(
            num_layers=2,
            num_local_layers=2,
            num_global_layers=2,
        )

        legacy_state = legacy.state_dict()
        explicit_state = explicit.state_dict()
        self.assertEqual(list(legacy_state), list(explicit_state))
        for name in legacy_state:
            self.assertTrue(torch.equal(legacy_state[name], explicit_state[name]), name)

        legacy.eval()
        explicit.eval()
        edges = {
            DRUG_RESPONSE: torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            CELL_SIMILARITY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            DRUG_SIMILARITY: torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        }
        torch.manual_seed(23)
        drug_features = torch.randn(3, 4)
        omics = {"genomics": {"mutation": torch.randn(2, 4)}}
        with torch.no_grad():
            legacy_output = legacy(
                drug_feature=drug_features,
                drug_adj=None,
                ibatch=None,
                omics_data=omics,
                hetero_graph_edge_index_dict=edges,
            )
            explicit_output = explicit(
                drug_feature=drug_features,
                drug_adj=None,
                ibatch=None,
                omics_data=omics,
                hetero_graph_edge_index_dict=edges,
            )
        for output_name in (
            "drug_embeddings",
            "cell_embeddings",
        ):
            self.assertTrue(
                torch.equal(legacy_output[output_name], explicit_output[output_name]),
                output_name,
            )
        for node_type in ("drug", "cell"):
            self.assertTrue(
                torch.equal(
                    legacy_output["fusion_weights"][node_type],
                    explicit_output["fusion_weights"][node_type],
                ),
                node_type,
            )

    def test_global_depth_can_change_without_changing_local_depth(self):
        model = self._build_model(
            num_layers=2,
            num_local_layers=2,
            num_global_layers=3,
        )
        self.assertEqual(len(model.local_convs), 2)
        self.assertEqual(len(model.global_convs), 3)
        self.assertEqual(len(model.global_norms), 3)

    def test_zero_global_depth_creates_local_only_model(self):
        model = self._build_model(num_local_layers=2, num_global_layers=0)
        self.assertEqual(len(model.local_convs), 2)
        self.assertEqual(len(model.global_convs), 0)
        self.assertTrue(model.use_local_branch)
        self.assertFalse(model.use_global_branch)
        self.assertIsNone(model.fusion)

    def test_zero_local_depth_creates_hgt_only_model(self):
        model = self._build_model(num_local_layers=0, num_global_layers=2)
        self.assertEqual(len(model.local_convs), 0)
        self.assertEqual(len(model.global_convs), 2)
        self.assertFalse(model.use_local_branch)
        self.assertTrue(model.use_global_branch)
        self.assertIsNone(model.fusion)

    def test_disabling_both_graph_branches_is_rejected(self):
        with self.assertRaises(ValueError):
            self._build_model(num_local_layers=0, num_global_layers=0)

    def test_local_only_forward_uses_local_embeddings_directly(self):
        model = self._build_model(num_local_layers=2, num_global_layers=0)
        model.eval()
        edges = {
            DRUG_RESPONSE: torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            CELL_SIMILARITY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            DRUG_SIMILARITY: torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        }
        with torch.no_grad():
            output = model(
                drug_feature=torch.randn(3, 4),
                drug_adj=None,
                ibatch=None,
                omics_data={"genomics": {"mutation": torch.randn(2, 4)}},
                hetero_graph_edge_index_dict=edges,
            )
        self.assertIsNone(output["global_embeddings"])
        self.assertEqual(output["fusion_weights"], {})
        self.assertTrue(
            torch.equal(
                output["node_embeddings"]["cell"],
                output["local_embeddings"]["cell"],
            )
        )

    def test_hgt_only_forward_uses_global_embeddings_directly(self):
        model = self._build_model(num_local_layers=0, num_global_layers=2)
        model.eval()
        edges = {
            DRUG_RESPONSE: torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            CELL_SIMILARITY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            DRUG_SIMILARITY: torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        }
        with torch.no_grad():
            output = model(
                drug_feature=torch.randn(3, 4),
                drug_adj=None,
                ibatch=None,
                omics_data={"genomics": {"mutation": torch.randn(2, 4)}},
                hetero_graph_edge_index_dict=edges,
            )
        self.assertIsNone(output["local_embeddings"])
        self.assertEqual(output["fusion_weights"], {})
        self.assertTrue(
            torch.equal(
                output["node_embeddings"]["cell"],
                output["global_embeddings"]["cell"],
            )
        )


class DirectedDistanceTests(unittest.TestCase):
    def setUp(self):
        self.drug_ids = ["D0", "D1", "D2", "D3"]
        self.cell_ids = ["C0", "C1", "C2"]
        self.edge_index_dict = {
            DRUG_SIMILARITY: torch.tensor(
                [
                    [0, 2],
                    [1, 0],
                ],
                dtype=torch.long,
            ),
            DRUG_RESPONSE: torch.tensor(
                [
                    [1],
                    [0],
                ],
                dtype=torch.long,
            ),
            CELL_SIMILARITY: torch.tensor(
                [
                    [0, 1],
                    [1, 2],
                ],
                dtype=torch.long,
            ),
        }

    def test_exact_directed_distances(self):
        pairs = pd.DataFrame(
            [
                {"drug_id": "D0", "cell_id": "C0", "label": 1, "prediction": 0.9},
                {"drug_id": "D0", "cell_id": "C1", "label": 1, "prediction": 0.8},
                {"drug_id": "D0", "cell_id": "C2", "label": 0, "prediction": 0.2},
                {"drug_id": "D3", "cell_id": "C2", "label": 0, "prediction": 0.1},
            ]
        )
        annotated = annotate_pair_distances(
            pairs,
            drug_ids=self.drug_ids,
            cell_ids=self.cell_ids,
            edge_index_dict=self.edge_index_dict,
            max_distance=3,
        )
        observed = {
            (row.drug_id, row.cell_id): (row.directed_distance, row.distance_bucket)
            for row in annotated.itertuples(index=False)
        }
        self.assertEqual(observed[("D0", "C0")], (2, "exact_2_hop"))
        self.assertEqual(observed[("D0", "C1")], (3, "exact_3_hop"))
        self.assertTrue(pd.isna(observed[("D0", "C2")][0]))
        self.assertEqual(observed[("D0", "C2")][1], "other")
        self.assertTrue(pd.isna(observed[("D3", "C2")][0]))
        self.assertEqual(observed[("D3", "C2")][1], "other")

    def test_similarity_edges_are_not_implicitly_reversed(self):
        pairs = pd.DataFrame(
            [{"drug_id": "D1", "cell_id": "C2", "label": 1, "prediction": 0.7}]
        )
        annotated = annotate_pair_distances(
            pairs,
            drug_ids=self.drug_ids,
            cell_ids=self.cell_ids,
            edge_index_dict=self.edge_index_dict,
            max_distance=3,
        )
        self.assertEqual(int(annotated.iloc[0]["directed_distance"]), 3)

    def test_evaluation_response_edge_leakage_is_rejected(self):
        leaked_pair = pd.DataFrame(
            [{"drug_id": "D1", "cell_id": "C0", "label": 1}]
        )
        with self.assertRaises(ValueError):
            assert_response_edges_exclude_pairs(
                leaked_pair,
                drug_ids=self.drug_ids,
                cell_ids=self.cell_ids,
                response_edge_index=self.edge_index_dict[DRUG_RESPONSE],
            )

    def test_non_response_pair_passes_leakage_check(self):
        held_out_pair = pd.DataFrame(
            [{"drug_id": "D0", "cell_id": "C2", "label": 1}]
        )
        assert_response_edges_exclude_pairs(
            held_out_pair,
            drug_ids=self.drug_ids,
            cell_ids=self.cell_ids,
            response_edge_index=self.edge_index_dict[DRUG_RESPONSE],
        )


if __name__ == "__main__":
    unittest.main()
