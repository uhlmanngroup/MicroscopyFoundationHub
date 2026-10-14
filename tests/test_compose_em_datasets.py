import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/data/compose_em_datasets.py"
spec = importlib.util.spec_from_file_location("compose_em_datasets", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_lucchi_pairs_by_numeric_id_not_lexical_position():
    images = [Path("mask0000.png"), Path("mask0001.png"), Path("mask0002.png")]
    masks = [Path("0.png"), Path("1.png"), Path("10.png"), Path("2.png")]

    with pytest.raises(ValueError, match="masks without images"):
        module.pair_lucchi(images, masks)

    pairs = module.pair_lucchi(images, masks[:2] + masks[3:])
    assert [(image.name, mask.name) for image, mask in pairs] == [
        ("mask0000.png", "0.png"),
        ("mask0001.png", "1.png"),
        ("mask0002.png", "2.png"),
    ]


def test_lucchi_missing_mask_fails():
    with pytest.raises(ValueError, match="Missing Lucchi mask"):
        module.pair_lucchi([Path("mask0002.png")], [Path("10.png")])


def test_staging_rejects_existing_wrong_symlink(tmp_path):
    correct = tmp_path / "2.png"
    wrong = tmp_path / "10.png"
    correct.touch()
    wrong.touch()
    staged = tmp_path / "staged.png"
    staged.symlink_to(wrong)

    with pytest.raises(RuntimeError, match="wrong source"):
        module.stage(correct, staged)
