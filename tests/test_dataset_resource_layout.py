"""Declared ICFG annotation filenames agree in preflight and actual loading."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import pytest

from datasets.icfgpedes import ICFGPEDES
from tools.check_resources import inspect_resources
from utils.options import get_args


@pytest.fixture
def layout(tmp_path):
    folder = tmp_path / "ICFG-PEDES"
    images = folder / "imgs"
    images.mkdir(parents=True)
    entries = [{"id": 10, "split": "train", "file_path": "train.png", "captions": ["a red coat"]},
               {"id": 20, "split": "test", "file_path": "test.png", "captions": ["a blue coat"]}]
    for name in ("train.png", "test.png"):
        Image.new("RGB", (8, 16)).save(images / name)
    bpe, weights = tmp_path / "vocab.gz", tmp_path / "clip.pt"
    bpe.write_bytes(b"resource-presence fixture")
    weights.write_bytes(b"resource-presence fixture")
    flags = ["--dataset_name", "ICFG-PEDES", "--bpe_path", str(bpe), "--clip_weights", str(weights)]
    return folder, entries, flags


def preflight(flags, root):
    with patch("tools.check_resources.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="fixture GPU")), \
            patch("tools.check_resources.importlib.util.find_spec", return_value=object()):
        return inspect_resources(get_args(flags + ["--root_dir", str(root)]))


@pytest.mark.parametrize("filename", ("ICFG-PEDES.json", "ICFG_PEDES.json"))
def test_primary_and_alias_work_from_parent_and_explicit_dataset_root(layout, filename):
    folder, entries, flags = layout
    annotation = folder / filename
    annotation.write_text(json.dumps(entries), encoding="utf-8")
    for root in (folder.parent, folder):
        dataset = ICFGPEDES(root=root, verbose=False)
        report = preflight(flags, root)
        assert report["ready"]
        assert Path(dataset.anno_path) == annotation
        assert report["resources"]["annotations"] == {"path": str(annotation), "exists": True}
        assert dataset.label_map == {10: 0} and dataset.test["image_pids"] == [20]


def test_identical_duplicates_prefer_primary_and_conflicting_duplicates_fail_consistently(layout):
    folder, entries, flags = layout
    primary, alias = folder / "ICFG-PEDES.json", folder / "ICFG_PEDES.json"
    content = json.dumps(entries)
    primary.write_text(content, encoding="utf-8")
    alias.write_text(content, encoding="utf-8")
    assert Path(ICFGPEDES(root=folder.parent, verbose=False).anno_path) == primary
    assert preflight(flags, folder.parent)["resources"]["annotations"]["path"] == str(primary)
    entries[0]["captions"] = ["a different authoritative description"]
    alias.write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(ValueError, match="Ambiguous ICFG-PEDES.*different contents"):
        ICFGPEDES(root=folder.parent, verbose=False)
    with pytest.raises(ValueError, match="Ambiguous ICFG-PEDES.*different contents"):
        preflight(flags, folder.parent)


def test_missing_annotations_keep_primary_in_resource_report(layout):
    folder, entries, flags = layout
    report = preflight(flags, folder.parent)
    assert not report["ready"] and report["missing_resources"] == ["annotations"]
    assert report["resources"]["annotations"]["path"] == str(folder / "ICFG-PEDES.json")
    with pytest.raises(FileNotFoundError, match="ICFG-PEDES.json"):
        ICFGPEDES(root=folder.parent, verbose=False)
