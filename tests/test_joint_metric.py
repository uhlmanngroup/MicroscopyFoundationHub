"""Small CPU-only checks for the scientific invariants and one complete pipeline."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from dino_peft.config import load_config
from dino_peft.analysis.joint_cache import _AlignedDataset, _resolve_sources, extract_joint_cache, occupancy_on_grid
from dino_peft.analysis.joint_metric import MetricConfig, SinkhornDistance, compute_metric, fit_barycenter, risk_from_components, sample_indices, _transform_distributions
from dino_peft.analysis.joint_run import recompute, resolve_experiment, run_one
from dino_peft.backbones.base import BackboneOutput


@pytest.fixture
def cfg():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield MetricConfig(l2_normalize=False, min_samples_per_distribution=1,
                       n_samples_per_distribution=16, barycenter_support=2,
                       barycenter_max_iterations=40, barycenter_tolerance=1e-7)
    torch.set_num_threads(before)


def clouds(points, separation=1.):
    result = {}
    for i, point in enumerate(points):
        foreground = np.tile(point, (4, 1)).astype(float)
        offset = np.zeros(foreground.shape[1])
        offset[-1] = separation
        result[str(i)] = {'foreground': foreground, 'background': foreground + offset}
    return result


def test_balanced_sampling_reproducibility_and_order(cfg):
    q = {'a': np.r_[np.ones(40), np.zeros(30), [.5]], 'b': np.r_[np.ones(9), np.zeros(8)]}
    a, b = sample_indices(q, cfg), sample_indices(dict(reversed(list(q.items()))), cfg)
    assert a['balanced_sample_count'] == 8
    assert a['counts']['a']['n_ambiguous'] == 1
    for dataset in q:
        for kind in ('foreground', 'background'):
            np.testing.assert_array_equal(a['indices'][dataset][kind], b['indices'][dataset][kind])
            assert len(np.unique(a['indices'][dataset][kind])) == 8
    with pytest.raises(ValueError, match='background'):
        sample_indices({'a': np.ones(8), 'b': np.ones(8)}, cfg)


def test_sinkhorn_cost_and_normalized_mass(cfg):
    distance = SinkhornDistance(cfg).distance
    assert distance(np.array([[0., 0.]]), np.array([[3., 4.]])) == pytest.approx(25.)
    x = np.array([[-1., 0.], [1., 0.], [.2, .5]])
    y = x + [.5, .3]
    assert distance(x, x) == pytest.approx(0., abs=1e-10)
    assert distance(x, y) == pytest.approx(distance(y, x), abs=1e-10)
    assert distance(np.tile(x, (7, 1)), y) == pytest.approx(distance(x, y), abs=1e-9)


def test_barycenter_equal_dataset_weights_and_multiple_modes(cfg):
    unequal = fit_barycenter({'a': np.zeros((2, 1)), 'b': np.full((50, 1), 6.)}, cfg)
    np.testing.assert_allclose(unequal['support'], 3., atol=1e-6)
    assert unequal['diagnostics']['dataset_weights'] == [.5, .5]
    bimodal = np.array([[-1.], [1.]])
    result = fit_barycenter({'a': bimodal, 'b': bimodal}, cfg)
    np.testing.assert_allclose(np.sort(result['support'], axis=0), bimodal, atol=1e-6)


def test_stable_selective_global_patterns_and_permutation(cfg):
    stable = compute_metric(clouds([[2., 2.]] * 3), cfg)
    assert stable['collection']['G'] < 1e-8 and stable['collection']['L'] < 1e-8
    data = clouds([[0., 0.], [0., 0.], [3., 0.]])
    selective = compute_metric(data, cfg)
    assert selective == compute_metric(dict(reversed(list(data.items()))), cfg)
    assert selective['per_dataset'][2]['r_i'] > 3.9 * selective['per_dataset'][0]['r_i']
    global_risk = compute_metric(clouds([[1., 0.], [-.5, np.sqrt(3)/2], [-.5, -np.sqrt(3)/2]], .2), cfg)
    assert global_risk['collection']['G'] > 20 and global_risk['collection']['L'] < 1e-4


def test_risk_population_std_and_separability():
    assert risk_from_components([1, 3], [0, 0], epsilon=1) == {'r': [1., 3.], 'G': 2., 'L': 1., 'L_rel': 1/3}
    assert risk_from_components([1, 1], [.1, 1])['r'][0] > 9 * risk_from_components([1, 1], [.1, 1])['r'][1]
    assert risk_from_components([1, 3], [10, 100], formula='consensus')['r'] == [1, 3]


def test_shared_balanced_pca(cfg):
    data = clouds([[1., 0., 0.], [0., 2., 0.]])
    data['0'] = {k: np.tile(v, (5, 1)) for k, v in data['0'].items()}
    transformed, meta = _transform_distributions(data, replace(cfg, pca_dim=2))
    np.testing.assert_allclose(meta['pca_mean'], [.5, 1., .5])
    for name, classes in data.items():
        for kind, array in classes.items():
            expected = (array - meta['pca_mean']) @ np.array(meta['pca_components']).T
            np.testing.assert_allclose(transformed[name][kind], expected)


def test_fractional_mask_alignment(tmp_path):
    images, masks = tmp_path / 'images', tmp_path / 'masks'
    images.mkdir(); masks.mkdir()
    mask = np.zeros((8, 8), np.uint8); mask[:, 2] = 255
    for directory in (images, masks):
        Image.fromarray(mask).save(directory / '0.png')
    source = _resolve_sources({'datasets': [{'id': 'a', 'image_dir': str(images), 'mask_dir': str(masks)}],
                               'preprocessing': {'center_crop_size': 4, 'img_size': [2, 2]}})[0]
    image, occupancy, _ = _AlignedDataset(source)[0]
    assert image.shape == (3, 2, 2)
    np.testing.assert_allclose(occupancy, [[.5, 0], [.5, 0]])
    np.testing.assert_allclose(occupancy_on_grid(occupancy, (1, 1), patch_size=2), [.25])


def test_cache_to_report_and_risk_only_reuse(tmp_path, cfg, monkeypatch):
    class Backbone(torch.nn.Module):
        patch_size = 2
        def forward(self, x):
            assert not self.training and not torch.is_grad_enabled()
            z = torch.nn.functional.avg_pool2d(x, 2).flatten(2).transpose(1, 2)
            return BackboneOutput(z.mean(1), z, (4, 4))
    datasets = []
    for name in ['a', 'b']:
        path = tmp_path / name; path.mkdir()
        mask = np.zeros((8, 8), np.uint8); mask[:, :4] = 255
        Image.fromarray(mask).save(path / '0.png')
        datasets.append({'id': name, 'image_dir': str(path), 'mask_dir': str(path)})
    experiment = {'cache_dir': str(tmp_path / 'cache'), 'results_root': str(tmp_path / 'results'),
                  'collection_id': 'synthetic', 'backbone_id': 'test', 'backbone': {'name': 'dinov3'},
                  'datasets': datasets, 'preprocessing': {'img_size': {'mode': 'native'}},
                  'extraction': {'device': 'cpu', 'preview_samples': 1}, 'metric': vars(cfg)}
    extract_joint_cache(experiment, backbone_factory=lambda *a, **kw: Backbone())
    def forbidden(*a, **kw):
        raise AssertionError('Reuse must not call backbone inference or OT')
    extract_joint_cache(experiment, backbone_factory=forbidden)
    output = run_one(experiment)
    assert all((output / p).is_file() for p in ['samples.json', 'components.json', 'index.html', 'per_dataset.csv', 'figs/components.svg'])
    result = json.loads((output / 'components.json').read_text())
    assert result['collection']['G'] < 1e-8
    monkeypatch.setattr('dino_peft.analysis.joint_run.compute_metric', forbidden)
    assert run_one(experiment) == output
    recompute(output / 'components.json', tmp_path / 'alternative', formula='consensus')
    with (tmp_path / 'cache/a/0000000.npz').open('ab') as f:
        f.write(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        run_one(experiment)


def test_shared_config_and_cluster_dependencies(tmp_path):
    config = load_config('configs/cluster/joint_metric.yaml')
    for collection in config['collections']:
        for backbone in config['backbones']:
            experiment = resolve_experiment(config, collection, backbone)
            sources = _resolve_sources(experiment)
            assert len(sources) == 3
            assert all(s['preprocessing']['binarize_threshold'] == (0 if collection == 'deepbacs' else 128) for s in sources)
            assert sources[0]['preprocessing']['img_size']['mode'] == ('native' if collection == 'deepbacs' else 'longest_edge')
    # Dry-run never calls a scheduler, but exercises the exact submission builder.
    run = subprocess.run([sys.executable, 'scripts/joint_metric.py', 'submit', '--collection', 'all', '--backbone', 'all', '--dry-run'], check=True, text=True, capture_output=True)
    assert run.stdout.count('--parsable') == 13
    assert '--dependency=afterok:900000' in run.stdout
    assert '--dependency=afterany:900001:900003:900005:900007:900009:900011' in run.stdout
