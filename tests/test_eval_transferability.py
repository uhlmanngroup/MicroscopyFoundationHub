"""Inference matrix and seed aggregation without GPU dependencies."""

import math
import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval_transferability import (
    aggregate, build_jobs, compose_target_config, missing_jobs, result_path,
    summarize_results,
)


def rows_for(jobs):
    return [
        {
            "modality": job.modality,
            "source": job.source,
            "target": target,
            "rep": job.rep,
            "seed": job.rep,
            "checkpoint": str(job.checkpoint),
            "n_images": 8,
            "foreground_iou": 0.1 * job.rep,
            "foreground_dice": 0.2 * job.rep,
            "mean_iou": 0.1 * job.rep,
            "mean_dice": 0.2 * job.rep,
        }
        for job in jobs
        for target in job.targets
    ]


class TransferabilityTests(unittest.TestCase):
    def test_all_directed_pairs_and_five_models(self):
        with tempfile.TemporaryDirectory() as root:
            for modality, datasets, fragment in (
                ("em", {"droso", "lucchi", "kasthuri"}, "dinov3_frozen_seg_rep"),
                ("deepbacs", {"coli", "aureus", "subtilis"}, "dinov3_head_seg_rep"),
            ):
                jobs = build_jobs(modality, range(1, 6), False, Path(root))
                self.assertEqual(len(jobs), 15)
                self.assertEqual(
                    {(job.source, target, job.rep) for job in jobs for target in job.targets},
                    {(source, target, rep) for source in datasets
                     for target in datasets - {source} for rep in range(1, 6)},
                )
                for job in jobs:
                    self.assertTrue(job.checkpoint.is_relative_to(root))
                    self.assertEqual(job.checkpoint.name, "best_model.pt")
                    self.assertIn(fragment + str(job.rep), str(job.checkpoint))

    def test_diagonal_is_opt_in(self):
        jobs = build_jobs("em", [1, 2], True, Path("/tmp/runs"))
        self.assertEqual(len(jobs), 6)
        self.assertTrue(all(set(job.targets) == {"droso", "lucchi", "kasthuri"} for job in jobs))

    def test_sample_standard_deviation_and_completeness(self):
        jobs = build_jobs("em", [1, 2, 3], False, Path("/tmp/runs"))
        rows = rows_for(jobs)
        summary = summarize_results(rows, jobs)
        self.assertEqual(len(summary), 6)
        for cell in summary:
            self.assertEqual(cell["n"], 3)
            self.assertEqual(cell["n_images"], 8)
            self.assertTrue(math.isclose(cell["foreground_iou_mean"], 0.2))
            self.assertTrue(math.isclose(cell["foreground_iou_std"], 0.1))
            self.assertTrue(math.isclose(cell["foreground_dice_mean"], 0.4))
            self.assertTrue(math.isclose(cell["foreground_dice_std"], 0.2))
        with self.assertRaises(ValueError):
            summarize_results(rows[:-1], jobs)
        with self.assertRaises(ValueError):
            summarize_results(rows + [rows[0]], jobs)
        inconsistent = [dict(row) for row in rows]
        inconsistent[0]["n_images"] = 7
        with self.assertRaises(ValueError):
            summarize_results(inconsistent, jobs)
        duplicate_seed = [dict(row) for row in rows]
        duplicate_seed[0]["seed"] = 2
        with self.assertRaises(ValueError):
            summarize_results(duplicate_seed, jobs)

    def test_target_data_rules_stay_separate_from_source_model_preprocessing(self):
        target = {
            "modality": "deepbacs", "num_classes": 2,
            "dataset": {"type": "paired", "params": {"pair_mode": "stem"}},
            "center_crop_size": 448, "binarize_threshold": 0,
            "img_size": {"mode": "native"}, "clahe_norm": False,
        }
        source = {
            "num_classes": 2, "img_size": {"mode": "native"},
            "clahe_norm": True, "binarize_threshold": 128,
        }
        result = compose_target_config(target, source, Path("/test/images"), Path("/test/masks"), "deepbacs")
        self.assertEqual(result["test_img_dir"], "/test/images")
        self.assertEqual(result["test_mask_dir"], "/test/masks")
        self.assertEqual(result["dataset"], target["dataset"])
        self.assertEqual(result["binarize_threshold"], 0)
        self.assertEqual(result["center_crop_size"], 448)
        self.assertTrue(result["clahe_norm"])
        self.assertEqual(target["clahe_norm"], False)

    def test_aggregate_writes_long_and_summary_tables(self):
        with tempfile.TemporaryDirectory() as root:
            out_dir = Path(root)
            jobs = build_jobs("em", [1, 2], False, out_dir / "checkpoints")
            for row in rows_for(jobs):
                job = next(job for job in jobs if job.source == row["source"] and job.rep == row["rep"])
                path = result_path(out_dir, job, row["target"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row))
            aggregate(out_dir, jobs)
            with (out_dir / "transfer_runs.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 12)
            with (out_dir / "transfer_summary.csv").open(newline="") as handle:
                cells = list(csv.DictReader(handle))
            self.assertEqual(len(cells), 6)
            self.assertTrue(all(cell["n"] == "2" for cell in cells))
            with (out_dir / "transfer_matrix_em_foreground_iou.csv").open(newline="") as handle:
                matrix = list(csv.DictReader(handle))
            self.assertEqual(len(matrix), 3)
            self.assertEqual(matrix[0][matrix[0]["source"]], "")

    def test_missing_jobs_report_whole_and_partial_inference_tasks(self):
        with tempfile.TemporaryDirectory() as root:
            out_dir = Path(root)
            jobs = build_jobs("em", [1, 2], False, out_dir / "checkpoints")
            for job in jobs[2:]:
                for target in job.targets:
                    path = result_path(out_dir, job, target)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}")
            first_target = jobs[1].targets[0]
            partial_path = result_path(out_dir, jobs[1], first_target)
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text("{}")

            missing = missing_jobs(out_dir, jobs)
            self.assertEqual([(index, targets) for index, _, targets in missing],
                             [(0, jobs[0].targets), (1, jobs[1].targets[1:])])
            with self.assertRaisesRegex(FileNotFoundError, "array indices: 0,1"):
                aggregate(out_dir, jobs)


if __name__ == "__main__":
    unittest.main()
