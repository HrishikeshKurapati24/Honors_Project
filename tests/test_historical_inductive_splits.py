import json
import os
import sys
import tempfile
import unittest

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmarking_common.splits import (  # noqa: E402
    PROTOCOL_UNSEEN_BOTH,
    PROTOCOL_UNSEEN_CELLS,
    PROTOCOL_UNSEEN_DRUGS,
    create_historical_protocol_folds,
    create_protocol_folds,
    ensure_historical_protocol_folds,
    save_historical_protocol_folds,
)


class HistoricalInductiveSplitTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for cell_index in range(20):
            for drug_index in range(10):
                rows.append(
                    {
                        "cell_id": f"C{cell_index:02d}",
                        "drug_id": f"D{drug_index:02d}",
                        "label": (cell_index + drug_index) % 2,
                    }
                )
        self.pairs = pd.DataFrame(rows)

    def test_historical_inductive_protocols_are_deterministic_and_leakage_free(self):
        for protocol in (
            PROTOCOL_UNSEEN_CELLS,
            PROTOCOL_UNSEEN_DRUGS,
            PROTOCOL_UNSEEN_BOTH,
        ):
            first = create_historical_protocol_folds(self.pairs, protocol=protocol)
            second = create_historical_protocol_folds(self.pairs, protocol=protocol)
            for first_fold, second_fold in zip(first, second):
                for split_name in ("train", "val", "test"):
                    pd.testing.assert_frame_equal(
                        first_fold[split_name],
                        second_fold[split_name],
                    )

                entities = first_fold["entities"]
                if protocol in {PROTOCOL_UNSEEN_CELLS, PROTOCOL_UNSEEN_BOTH}:
                    self._assert_three_way_disjoint(
                        entities["train_cells"],
                        entities["val_cells"],
                        entities["test_cells"],
                    )
                if protocol in {PROTOCOL_UNSEEN_DRUGS, PROTOCOL_UNSEEN_BOTH}:
                    self._assert_three_way_disjoint(
                        entities["train_drugs"],
                        entities["val_drugs"],
                        entities["test_drugs"],
                    )

    def test_historical_validation_entities_differ_from_newer_policy(self):
        for protocol in (
            PROTOCOL_UNSEEN_CELLS,
            PROTOCOL_UNSEEN_DRUGS,
            PROTOCOL_UNSEEN_BOTH,
        ):
            historical = create_historical_protocol_folds(self.pairs, protocol=protocol)
            newer = create_protocol_folds(self.pairs, protocol=protocol)
            self.assertTrue(
                any(
                    historical[index]["entities"] != newer[index]["entities"]
                    for index in range(5)
                ),
                protocol,
            )
            for index in range(5):
                pd.testing.assert_frame_equal(
                    historical[index]["test"],
                    newer[index]["test"],
                )

    def test_ensure_rejects_existing_split_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            response_path = os.path.join(temp_dir, "response_pairs.csv")
            split_dir = os.path.join(temp_dir, "splits")
            self.pairs.to_csv(response_path, index=False)
            save_historical_protocol_folds(
                self.pairs,
                split_dir,
                protocol=PROTOCOL_UNSEEN_CELLS,
            )

            train_path = os.path.join(split_dir, "fold_1", "train.csv")
            drifted = pd.read_csv(train_path).iloc[1:].reset_index(drop=True)
            drifted.to_csv(train_path, index=False)

            with self.assertRaisesRegex(ValueError, "does not match the historical split"):
                ensure_historical_protocol_folds(
                    response_pairs_path=response_path,
                    output_dir=split_dir,
                    protocol=PROTOCOL_UNSEEN_CELLS,
                )

    def test_historical_manifest_records_seed_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_historical_protocol_folds(
                self.pairs,
                temp_dir,
                protocol=PROTOCOL_UNSEEN_BOTH,
            )
            with open(os.path.join(temp_dir, "split_manifest.json")) as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["split_generator"], "historical_inductive_v1")
            self.assertEqual(
                manifest["validation_seed_policy"],
                "fixed_base_seed_per_outer_fold",
            )

    def _assert_three_way_disjoint(self, train, val, test):
        self.assertFalse(set(train) & set(val))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(val) & set(test))


if __name__ == "__main__":
    unittest.main()
