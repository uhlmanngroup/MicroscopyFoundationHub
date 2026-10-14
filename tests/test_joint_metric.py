"""Small CPU checks for local geometry, probe leakage and cache reuse."""
import json
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from dino_peft.analysis.joint_cache import extract_joint_cache, load_cache_manifest
from dino_peft.analysis.joint_metric import (MetricConfig, direction_alignment, image_split,
    local_directions, near_foreground_background, run_linear_probes)
from dino_peft.analysis.joint_run import run_one
from dino_peft.analysis.joint_report import summarize_results
from dino_peft.backbones.base import BackboneOutput
from dino_peft.config import load_config


def test_local_background_never_crosses_images_and_skips_interiors():
    config = MetricConfig(near_foreground_radius_patches=1, l2_normalize=False)
    q = np.array([1, 1, 1, 0,
                  1, 1, 1, 0,
                  1, 1, 1, 0,
                  0, 0, 0, 0.], dtype=float)
    z = np.zeros((16, 2), dtype=np.float32)
    z[q == 1] = [2, 1]
    z[q == 0] = [1, 1]
    result = local_directions(z, q, (4, 4), config)
    assert result["n_foreground_all"] == 9
    assert result["n_skipped_no_local_background"] == 4
    np.testing.assert_array_equal(result["vectors"], np.tile([1, 0], (5, 1)))
    near = near_foreground_background(q, (4, 4), tau_fg=.8, tau_bg=.2, radius=1)
    assert np.all(near[result["background_indices"]])
    assert not near_foreground_background(np.zeros(16), (4, 4), tau_fg=.8, tau_bg=.2, radius=1).any()


def test_direction_alignment_is_symmetric_and_image_balanced():
    def image(direction, n):
        return {"mean_vector": np.array(direction), "mean_magnitude": float(np.linalg.norm(direction)),
                "mean_foreground": np.array(direction), "mean_background": np.zeros(2),
                "n_vectors": n, "n_skipped_no_local_background": 0}
    data = {"a": [image([1, 0], 1), image([1, 0], 100)],
            "b": [image([1, 0], 2)], "c": [image([-1, 0], 5)]}
    result = direction_alignment(data, MetricConfig())
    np.testing.assert_allclose(result["pairwise_alignment"], [[1, 1, -1], [1, 1, -1], [-1, -1, 1]])
    assert result["collection"]["A_collection"] == pytest.approx(-1 / 3)
    assert [r["A_i"] for r in result["per_dataset"]] == [0, 0, -1]
    assert result == direction_alignment(dict(reversed(list(data.items()))), MetricConfig())


def test_image_disjoint_probe_detects_opposing_label_directions():
    config = MetricConfig(l2_normalize=False, probe_max_iterations=35, probe_l2=.01)
    samples = {}
    for name, sign in (("a", 1), ("b", -1)):
        samples[name] = {}
        for index in range(6):
            samples[name][f"image-{index}"] = {
                "foreground": np.tile([float(sign), 0.], (4, 1)),
                "background": np.tile([0., 0.], (4, 1))}
    report, weights = run_linear_probes(samples, config)
    for name, groups in report["split"].items():
        assert set(groups["train"]).isdisjoint(groups["validation"])
        assert set(groups["train"]).isdisjoint(groups["test"])
        assert set(groups["validation"]).isdisjoint(groups["test"])
        assert len(set().union(*map(set, groups.values()))) == len(samples[name])
    for row in report["per_dataset"]:
        assert row["test_individual_balanced_accuracy"] == 1
        assert row["test_joint_balanced_accuracy"] == .5
        assert row["test_joint_minus_individual"] == -.5
    assert weights["individual_weights"].shape == (2, 2)
    assert image_split(list(samples["a"]), seed=config.seed, dataset="a",
                       train_fraction=.6, validation_fraction=.2) == report["split"]["a"]


def test_cache_to_direction_report_without_rerunning_backbone(tmp_path):
    class Backbone(torch.nn.Module):
        patch_size = 2
        def forward(self, x):
            assert not self.training and not torch.is_grad_enabled()
            z = torch.nn.functional.avg_pool2d(x, 2).flatten(2).transpose(1, 2)
            return BackboneOutput(z.mean(1), z, (4, 4))
    datasets = []
    for name in ("a", "b"):
        path = tmp_path / name
        path.mkdir()
        mask = np.zeros((8, 8), np.uint8)
        mask[:, :4] = 255
        for index in range(5):
            Image.fromarray(mask).save(path / f"{index}.png")
        datasets.append({"id": name, "image_dir": str(path), "mask_dir": str(path)})
    config = MetricConfig(l2_normalize=False, near_foreground_radius_patches=1,
                          probe_patches_per_image_per_class=4, probe_max_iterations=25)
    experiment = {"cache_dir": str(tmp_path / "cache"), "results_root": str(tmp_path / "results"),
                  "collection_id": "synthetic", "backbone_id": "test", "backbone": {"name": "dinov3"},
                  "datasets": datasets, "preprocessing": {"img_size": {"mode": "native"}},
                  "extraction": {"device": "cpu", "preview_samples": 1}, "metric": vars(config)}
    extract_joint_cache(experiment, backbone_factory=lambda *args, **kwargs: Backbone())
    def forbidden(*args, **kwargs):
        raise AssertionError("The cached run must not execute the backbone")
    extract_joint_cache(experiment, backbone_factory=forbidden)
    output = run_one(experiment)
    result = json.loads((output / "components.json").read_text())
    assert result["schema_version"] == 2
    assert result["collection"]["A_collection"] == pytest.approx(1)
    assert set(json.loads((output / "probe_split.json").read_text())) == {"a", "b"}
    assert all(row["n_skipped_no_local_background"] == 20 for row in result["per_dataset"])
    assert all(row["test_individual_balanced_accuracy"] == 1 for row in result["per_dataset"])
    assert all(row["test_joint_balanced_accuracy"] == 1 for row in result["per_dataset"])
    assert all((output / name).is_file() for name in ("local_vectors.npz", "probe_weights.npz",
           "probe_split.json", "pairwise_alignment.csv", "probe_results.csv", "cloud_projection.npz",
           "figs/alignment_matrix.svg", "figs/linear_probe.png", "figs/local_clouds.png"))
    with np.load(output / "local_vectors.npz") as data:
        assert len(data["vectors"]) == 40
        assert data["vectors"].shape[1] == 3
    html = (output / "index.html").read_text()
    assert "barycenter" not in html.lower() and "Global risk" not in html
    assert "src=\"data:image/png;base64," in html
    assert len(result["alignment"]["previews"]) == 6
    assert all(p["cache_occupancy_verified"] for p in result["alignment"]["previews"])
    legacy = tmp_path / "results" / "old" / "run"
    legacy.mkdir(parents=True)
    (legacy / "components.json").write_text(json.dumps({"schema_version": 1, "collection": {"G": 9}}))
    assert len(summarize_results(tmp_path / "results")) == 1
    assert "old/run" not in (tmp_path / "results" / "index.html").read_text()
    assert "G" not in (tmp_path / "results" / "all_collections.csv").read_text().splitlines()[0]
    assert run_one(experiment) == output
    manifest = load_cache_manifest(tmp_path / "cache")
    assert manifest["extraction_hash"] == result["cache_hash"]
    with (tmp_path / "cache/a/0000000.npz").open("ab") as file:
        file.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        run_one(experiment)


def test_shared_config_and_slurm_dry_run():
    cfg = load_config("configs/cluster/joint_metric.yaml")
    assert cfg["metric"]["near_foreground_radius_patches"] == 2
    assert "sinkhorn_scaling" not in cfg["metric"]
    completed = subprocess.run([sys.executable, "scripts/joint_metric.py", "submit", "--collection", "em",
                                "--backbone", "dinov3", "--metrics-only", "--dry-run"],
                               check=True, text=True, capture_output=True)
    assert "--job-name=joint-compute" in completed.stdout
    assert "--gpus=" not in completed.stdout
