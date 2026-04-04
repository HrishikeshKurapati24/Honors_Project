import os
import sys
import tempfile
import types
import unittest

import h5py
import hickle as hkl
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmarking_common.drug_features import copy_graph_subset, load_graph_feature
from benchmarking_common.data_prep import prepare_3omics_dataset1
from benchmarking_common.experiment import allowed_models_for_protocol
from benchmarking_common.metrics import compute_binary_metrics
from benchmarking_common.results import build_comparison_rows, load_best_config, save_model_outputs, save_tuning_outputs
from benchmarking_common import load_module_from_path
from benchmarking_common.splits import (
    PROTOCOL_UNSEEN_BOTH,
    PROTOCOL_UNSEEN_CELLS,
    PROTOCOL_UNSEEN_DRUGS,
    create_protocol_folds,
    create_soulcdr_folds,
)
from benchmarking_common.tuning import default_fixed_config, should_tune_random
from benchmark_wrappers.deepcdr_torch_runner import MAX_ATOMS
from benchmark_wrappers.deepcdr_torch_runner import _load_deepcdr_port
from benchmark_wrappers.deepcdr_torch_runner import _pad_graph as pad_graph


class BenchmarkingPipelineTest(unittest.TestCase):
    def test_soulcdr_split_reproducibility(self):
        response_pairs = pd.DataFrame(
            {
                "cell_id": ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4", "c5", "c5"],
                "drug_id": ["d1", "d2", "d1", "d2", "d1", "d2", "d1", "d2", "d1", "d2"],
                "label": [1, 0, 0, 1, 1, 0, 0, 1, 0, 1],
            }
        )
        first = create_soulcdr_folds(response_pairs, seed=0, n_splits=5)
        second = create_soulcdr_folds(response_pairs, seed=0, n_splits=5)
        self.assertEqual(
            [fold["train"].to_dict() for fold in first],
            [fold["train"].to_dict() for fold in second],
        )
        self.assertEqual(
            [fold["val"].to_dict() for fold in first],
            [fold["val"].to_dict() for fold in second],
        )
        self.assertEqual(
            [fold["test"].to_dict() for fold in first],
            [fold["test"].to_dict() for fold in second],
        )

    def test_binary_metrics_are_bounded(self):
        metrics = compute_binary_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.4, 0.8]))
        self.assertGreaterEqual(metrics["auc"], 0.0)
        self.assertLessEqual(metrics["auc"], 1.0)
        self.assertGreaterEqual(metrics["aupr"], 0.0)
        self.assertLessEqual(metrics["aupr"], 1.0)
        self.assertGreaterEqual(metrics["f1"], 0.0)
        self.assertLessEqual(metrics["f1"], 1.0)
        self.assertGreaterEqual(metrics["acc"], 0.0)
        self.assertLessEqual(metrics["acc"], 1.0)

    def test_results_aggregation_writes_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = save_model_outputs(
                model_results_dir=os.path.join(temp_dir, "dataset-1", "SOULCDR"),
                fold_metrics=[
                    {"fold": 1, "best_val_auc": 0.7, "auc": 0.8, "aupr": 0.75, "f1": 0.7, "acc": 0.72},
                    {"fold": 2, "best_val_auc": 0.8, "auc": 0.85, "aupr": 0.8, "f1": 0.75, "acc": 0.78},
                ],
                prediction_rows_by_fold={1: [], 2: []},
                metadata={"model": "SOULCDR"},
            )
            self.assertTrue(os.path.isfile(os.path.join(temp_dir, "dataset-1", "SOULCDR", "summary.json")))
            rows = build_comparison_rows(temp_dir)
            self.assertEqual(len(rows), 1)
            self.assertGreater(summary["mean"]["auc"], 0)

    def test_tuning_outputs_persist_best_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = {"model": "SOULCDR", "tuned": True, "config": {"lr": 1e-3}}
            save_tuning_outputs(
                model_results_dir=temp_dir,
                trials=[{"stage": 1, "candidate_id": 1, "auc": 0.8, "config": '{"lr": 0.001}'}],
                best_config_payload=payload,
            )
            self.assertEqual(load_best_config(temp_dir)["config"]["lr"], 1e-3)

    def test_unseen_cell_split_has_disjoint_cells(self):
        response_pairs = pd.DataFrame(
            {
                "cell_id": [f"c{i}" for i in range(6) for _ in range(3)],
                "drug_id": ["d1", "d2", "d3"] * 6,
                "label": [0, 1, 0, 1, 0, 1] * 3,
            }
        )
        folds = create_protocol_folds(response_pairs, protocol=PROTOCOL_UNSEEN_CELLS, seed=0, n_splits=3)
        for fold in folds:
            entities = fold["entities"]
            self.assertFalse(set(entities["train_cells"]) & set(entities["val_cells"]))
            self.assertFalse(set(entities["train_cells"]) & set(entities["test_cells"]))
            self.assertFalse(set(entities["val_cells"]) & set(entities["test_cells"]))

    def test_unseen_drug_split_has_disjoint_drugs(self):
        response_pairs = pd.DataFrame(
            {
                "cell_id": ["c1", "c2", "c3", "c4"] * 4,
                "drug_id": [f"d{i}" for i in range(4) for _ in range(4)],
                "label": [0, 1, 0, 1] * 4,
            }
        )
        folds = create_protocol_folds(response_pairs, protocol=PROTOCOL_UNSEEN_DRUGS, seed=0, n_splits=4)
        for fold in folds:
            entities = fold["entities"]
            self.assertFalse(set(entities["train_drugs"]) & set(entities["val_drugs"]))
            self.assertFalse(set(entities["train_drugs"]) & set(entities["test_drugs"]))
            self.assertFalse(set(entities["val_drugs"]) & set(entities["test_drugs"]))

    def test_unseen_both_test_pairs_are_fully_unseen(self):
        response_pairs = pd.DataFrame(
            {
                "cell_id": [f"c{i}" for i in range(6) for _ in range(6)],
                "drug_id": [f"d{j}" for _ in range(6) for j in range(6)],
                "label": [int((i + j) % 2 == 0) for i in range(6) for j in range(6)],
            }
        )
        folds = create_protocol_folds(response_pairs, protocol=PROTOCOL_UNSEEN_BOTH, seed=0, n_splits=3)
        for fold in folds:
            entities = fold["entities"]
            test_pairs = fold["test"]
            self.assertTrue(set(test_pairs["cell_id"]).issubset(set(entities["test_cells"])))
            self.assertTrue(set(test_pairs["drug_id"]).issubset(set(entities["test_drugs"])))

    def test_tuning_policy_matches_requested_matrix(self):
        self.assertTrue(should_tune_random("3OmicsBenchmarking", "dataset-1", "SOULCDR"))
        self.assertFalse(should_tune_random("3OmicsBenchmarking", "dataset-1", "GraphCDR"))
        self.assertTrue(should_tune_random("3OmicsBenchmarking", "dataset-2", "GraphCDR"))
        self.assertFalse(should_tune_random("ExpressionBenchmarking", "dataset-2", "SOULCDR"))
        self.assertTrue(should_tune_random("ExpressionBenchmarking", "dataset-2", "HRLCDR"))
        self.assertEqual(default_fixed_config("PathwayBenchmarking", "dataset-1", "GPDRP")["architecture"], "GIN_TRANSFORMER")

    def test_expression_unseen_both_excludes_hrlcdr(self):
        allowed = allowed_models_for_protocol(
            "ExpressionBenchmarking",
            "unseen_both",
            ["SOULCDR", "HRLCDR"],
        )
        self.assertEqual(allowed, ["SOULCDR"])

    def test_3omics_dataset1_preparation_keeps_graphcdr_drug_universe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = prepare_3omics_dataset1(ROOT_DIR, os.path.join(ROOT_DIR, "3OmicsBenchmarking"), temp_dir)
            self.assertEqual(metadata["drug_count"], 222)
            response_pairs = pd.read_csv(os.path.join(temp_dir, "response_pairs.csv"))
            self.assertEqual(response_pairs["drug_id"].nunique(), 222)

    def test_copy_graph_subset_skips_same_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph = [np.ones((2, 3), dtype=np.float32), [[1], [0]], [1, 1]]
            graph_path = os.path.join(temp_dir, "123.hkl")
            hkl.dump(graph, graph_path)
            copy_graph_subset(temp_dir, temp_dir, ["123"])
            self.assertTrue(os.path.isfile(graph_path))

    def test_load_graph_feature_repairs_string_type_attrs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph = (np.ones((2, 3), dtype=np.float32), [[1], [0]], [1, 1])
            graph_path = os.path.join(temp_dir, "broken.hkl")
            hkl.dump(graph, graph_path)

            with h5py.File(graph_path, "r+") as handle:
                def rewrite_type_attr(node):
                    if "type" not in node.attrs:
                        return
                    current = node.attrs["type"]
                    if isinstance(current, np.ndarray):
                        values = current.reshape(-1).tolist()
                    else:
                        values = [current]
                    string_values = [
                        value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
                        for value in values
                    ]
                    node.attrs.modify("type", np.asarray(string_values, dtype=object))

                rewrite_type_attr(handle)
                handle.visititems(lambda _name, obj: rewrite_type_attr(obj))

            feat_mat, adj_list, degree_list = load_graph_feature(graph_path)
            self.assertEqual(feat_mat.shape, (2, 3))
            self.assertEqual(adj_list, [[1], [0]])
            self.assertEqual(degree_list, [1, 1])

    def test_deepcdr_port_forward_backward(self):
        module = _load_deepcdr_port(ROOT_DIR)
        model = module.DeepCDRTorchModel(
            drug_dim=4,
            mutation_dim=1300,
            gexpr_dim=8,
            methy_dim=6,
            units_list=[8, 8, 8],
            use_mut=True,
            use_gexp=True,
            use_methy=True,
            use_relu=True,
            use_bn=True,
            use_gmp=True,
        )
        drug_feat = torch.randn(2, MAX_ATOMS, 4)
        drug_adj = torch.eye(MAX_ATOMS).repeat(2, 1, 1)
        mutation = torch.randn(2, 1, 1, 1300)
        gexpr = torch.randn(2, 8)
        methy = torch.randn(2, 6)
        labels = torch.tensor([0.0, 1.0])
        outputs = model(drug_feat, drug_adj, mutation, gexpr, methy)
        loss = torch.nn.BCELoss()(outputs, labels)
        loss.backward()
        self.assertEqual(tuple(outputs.shape), tuple(labels.shape))

    def test_deepcdr_graph_padding_shape(self):
        feat = np.ones((5, 4), dtype=np.float32)
        adj_list = [[1], [0, 2], [1, 3], [2, 4], [3]]
        padded_feat, padded_adj = pad_graph(feat, adj_list)
        self.assertEqual(padded_feat.shape, (MAX_ATOMS, 4))
        self.assertEqual(padded_adj.shape, (MAX_ATOMS, MAX_ATOMS))

    def test_hrlcdr_normalization_handles_constant_expression_columns(self):
        sys.modules.setdefault("seaborn", types.ModuleType("seaborn"))
        sys.modules.setdefault("pubchempy", types.ModuleType("pubchempy"))

        module_dir = os.path.join(ROOT_DIR, "benchmark models", "HRLCDR-master", "GDSC")
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

        utils_module = load_module_from_path("test_hrlcdr_utils", os.path.join(module_dir, "myutils.py"))
        models_module = load_module_from_path("test_hrlcdr_models", os.path.join(module_dir, "models.py"))

        expression = np.array(
            [
                [0.0, 1.0, 5.0, 2.0],
                [0.0, 3.0, 5.0, 4.0],
                [0.0, 5.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
        fingerprints = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        adjacency = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

        load_feature = models_module.LoadFeature(expression, fingerprints, device="cpu")
        cell_feat, drug_feat = load_feature()
        self.assertTrue(torch.isfinite(cell_feat).all())
        self.assertTrue(torch.isfinite(drug_feat).all())

        cell_sim = utils_module.k_near_graph(
            utils_module.calculate_gene_exponent_similarity7(torch.from_numpy(expression), mu=3),
            2,
        )
        drug_sim = utils_module.k_near_graph(utils_module.jaccard_coef7(torch.from_numpy(fingerprints)), 2)
        _, _, _, drug_hyper_raw, _, drug_hyper_full = utils_module.hyper_graph(drug_sim, adjacency.T, cell_sim)
        _, _, _, cell_hyper_raw, _, _ = utils_module.hyper_graph(cell_sim, adjacency, drug_sim)
        cell_hyper = cell_hyper_raw * (1.0 / cell_hyper_raw.sum(dim=1, keepdim=True).clamp_min(1e-8))
        drug_hyper = drug_hyper_full * (1.0 / drug_hyper_full.sum(dim=1, keepdim=True).clamp_min(1e-8))

        model = models_module.hrlcdr_new(
            adj_mat=adjacency,
            cell_exprs=expression,
            drug_finger=fingerprints,
            gamma=8.7,
            drug_hyper=drug_hyper,
            cell_hyper=cell_hyper,
            device="cpu",
            dim=8,
        )
        output = model()
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
