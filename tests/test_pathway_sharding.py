import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLEXIBLE_DIR = os.path.join(ROOT_DIR, "flexible model")
if FLEXIBLE_DIR not in sys.path:
    sys.path.insert(0, FLEXIBLE_DIR)

from data_flexible import list_available_omics  # noqa: E402
from flexibility_utils import (  # noqa: E402
    build_dataset_view,
    build_pathway_shards,
)
from model_flexible import (  # noqa: E402
    EncoderConfig,
    FlexibleCellLineRepresentationModule,
)


class PathwayShardingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_root = Path(self.temp_dir.name) / "dataset"
        self.dataset_root.mkdir()
        (self.dataset_root / "drug_graph_feat").mkdir()
        (self.dataset_root / "drug_graph_feat" / "placeholder.txt").write_text("graph")

        self.pathway = pd.DataFrame(
            {
                "p1": [1.0, 2.0],
                "p2": [3.0, 4.0],
                "p3": [5.0, 6.0],
                "p4": [7.0, 8.0],
                "p5": [9.0, 10.0],
                "p6": [11.0, 12.0],
            },
            index=pd.Index(["C1", "C2"], name="cell_id"),
        )
        self.pathway.to_csv(self.dataset_root / "pathway.csv")
        pd.DataFrame({"m1": [0, 1]}, index=["C1", "C2"]).to_csv(
            self.dataset_root / "genomics_mutation.csv"
        )
        pd.DataFrame(
            {"cell_id": ["C1", "C2"], "drug_id": ["D1", "D1"], "label": [0, 1]}
        ).to_csv(self.dataset_root / "response_pairs.csv", index=False)
        pd.DataFrame({"s1": [1.0, 0.0]}, index=["C1", "C2"]).to_csv(
            self.dataset_root / "similarity.csv"
        )
        pd.DataFrame({"f1": [1.0]}, index=["D1"]).to_csv(
            self.dataset_root / "physicochemical.csv"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empty_include_stems_copies_only_generated_shards(self):
        shards = build_pathway_shards(
            base_dataset_root=self.dataset_root,
            shard_count=2,
        )
        view_dir = Path(self.temp_dir.name) / "view"
        build_dataset_view(
            view_dir,
            base_dataset_root=self.dataset_root,
            include_stems=[],
            extra_tables=shards,
        )

        self.assertFalse((view_dir / "pathway.csv").exists())
        self.assertFalse((view_dir / "genomics_mutation.csv").exists())
        self.assertEqual(
            sorted(path.stem for path in view_dir.glob("pathway_shard_*.csv")),
            ["pathway_shard_01", "pathway_shard_02"],
        )
        manifest = json.loads((view_dir / "view_manifest.json").read_text())
        self.assertEqual(manifest["selected_stems"], [])
        self.assertEqual(
            manifest["extra_tables"],
            ["pathway_shard_01", "pathway_shard_02"],
        )

        available = list_available_omics(str(view_dir))
        self.assertEqual(
            [entry["stem"] for entry in available],
            ["pathway_shard_01", "pathway_shard_02"],
        )

    def test_shards_preserve_every_source_column_once(self):
        shards = build_pathway_shards(
            base_dataset_root=self.dataset_root,
            shard_count=4,
        )
        reconstructed = pd.concat([shards[stem] for stem in sorted(shards)], axis=1)
        pd.testing.assert_frame_equal(reconstructed, self.pathway)
        self.assertEqual([frame.shape[1] for frame in shards.values()], [2, 2, 1, 1])

    def test_more_shards_than_features_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            build_pathway_shards(
                base_dataset_root=self.dataset_root,
                shard_count=7,
            )

    def test_multiple_shards_activate_intra_pathway_fusion(self):
        parameter_counts = []
        for shard_count in (1, 2):
            shards = build_pathway_shards(
                base_dataset_root=self.dataset_root,
                shard_count=shard_count,
            )
            configs = [
                EncoderConfig(
                    category="pathway",
                    subtype=stem.removeprefix("pathway_"),
                    encoder_type="pathway_fc",
                    input_dim=frame.shape[1],
                    output_dim=8,
                )
                for stem, frame in sorted(shards.items())
            ]
            module = FlexibleCellLineRepresentationModule(
                configs,
                fusion_dim=8,
                output_dim=8,
            )
            parameter_counts.append(
                sum(parameter.numel() for parameter in module.parameters())
            )
            self.assertEqual(len(module.encoders), shard_count)
            self.assertEqual("pathway" in module.intra_fusion, shard_count > 1)

        self.assertGreater(parameter_counts[1], parameter_counts[0])

    def test_one_shard_is_numerically_equivalent_to_unsplit_pathway(self):
        baseline_config = EncoderConfig(
            category="pathway",
            subtype="pathway",
            encoder_type="pathway_fc",
            input_dim=self.pathway.shape[1],
            output_dim=8,
        )
        shard_config = EncoderConfig(
            category="pathway",
            subtype="shard_01",
            encoder_type="pathway_fc",
            input_dim=self.pathway.shape[1],
            output_dim=8,
        )

        torch.manual_seed(29)
        baseline = FlexibleCellLineRepresentationModule(
            [baseline_config],
            fusion_dim=8,
            output_dim=8,
        )
        torch.manual_seed(29)
        one_shard = FlexibleCellLineRepresentationModule(
            [shard_config],
            fusion_dim=8,
            output_dim=8,
        )
        baseline.eval()
        one_shard.eval()

        values = torch.from_numpy(self.pathway.to_numpy(dtype="float32", copy=True))
        with torch.no_grad():
            baseline_output = baseline({"pathway": {"pathway": values}})
            shard_output = one_shard({"pathway": {"shard_01": values}})

        self.assertEqual(
            sum(parameter.numel() for parameter in baseline.parameters()),
            sum(parameter.numel() for parameter in one_shard.parameters()),
        )
        self.assertTrue(torch.equal(baseline_output, shard_output))


if __name__ == "__main__":
    unittest.main()
