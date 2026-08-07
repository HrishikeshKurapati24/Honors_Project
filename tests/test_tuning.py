import os
import sys
import unittest


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmarking_common.tuning import should_tune_random, tuning_candidates  # noqa: E402


class TuningConfigurationTests(unittest.TestCase):
    def test_strict_models_are_enabled_for_both_datasets(self):
        for dataset in ("dataset-1", "dataset-2"):
            for model in (
                "FUSECDR",
                "FUSECDR_minibatch",
                "GraphCDR",
                "RedCDR",
                "GADRP",
                "DeepTTC",
                "GraphDRP",
            ):
                self.assertTrue(
                    should_tune_random("3OmicsStrictBenchmarking", dataset, model),
                    (dataset, model),
                )

    def test_tuning_candidate_grids_remain_complete(self):
        self.assertEqual(len(tuning_candidates("FUSECDR", "3OmicsStrictBenchmarking")), 9)
        self.assertEqual(len(tuning_candidates("GraphDRP", "3OmicsStrictBenchmarking")), 16)
        self.assertEqual(len(tuning_candidates("GraphDRP")), 6)


if __name__ == "__main__":
    unittest.main()
