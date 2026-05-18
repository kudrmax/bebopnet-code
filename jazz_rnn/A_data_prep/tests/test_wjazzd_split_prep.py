"""Tests for wjazzd_split_prep.route_split (3-bucket layout)."""
from __future__ import annotations

from pathlib import Path

import pytest

from jazz_rnn.A_data_prep.wjazzd_split_prep import route_split


@pytest.fixture()
def fake_xml_source(tmp_path: Path) -> Path:
    """Pool of 6 fake xml files: a, b, c, d, e, f."""
    src = tmp_path / "src"
    src.mkdir()
    for name in ("a", "b", "c", "d", "e", "f"):
        (src / f"{name}.xml").write_text(f"<dummy>{name}</dummy>")
    return src


def test_routes_each_bucket_to_distinct_directory(
    fake_xml_source: Path, tmp_path: Path
) -> None:
    out_root = tmp_path / "out"
    split = {"train": ["a", "b"], "eval": ["c", "d"], "test": ["e", "f"]}

    result = route_split(fake_xml_source, split, out_root)

    train_dir = out_root / "train" / "wjazzd"
    val_dir = out_root / "test" / "wjazzd"
    test_canonical_dir = out_root / "test_canonical" / "wjazzd"

    assert sorted(p.name for p in train_dir.iterdir()) == ["a.xml", "b.xml"]
    assert sorted(p.name for p in val_dir.iterdir()) == ["c.xml", "d.xml"]
    assert sorted(p.name for p in test_canonical_dir.iterdir()) == ["e.xml", "f.xml"]
    assert result.n_train == 2
    assert result.n_val == 2
    assert result.n_test_canonical == 2


def test_test_canonical_lives_outside_authorial_globs(
    fake_xml_source: Path, tmp_path: Path
) -> None:
    """gather_data_from_xml globs ``train/*/*.xml`` and ``test/*/*.xml``;
    test_canonical must NOT be under either prefix or it would leak in."""
    out_root = tmp_path / "out"
    split = {"train": ["a"], "eval": ["b"], "test": ["c"]}

    route_split(fake_xml_source, split, out_root)

    test_canonical = out_root / "test_canonical" / "wjazzd" / "c.xml"
    assert test_canonical.is_file()
    # Must not appear under "test/" — otherwise the author's `test/*/*.xml`
    # glob would suck it into val.pkl alongside split[eval].
    assert not (out_root / "test" / "wjazzd" / "c.xml").exists()
    assert not (out_root / "train" / "wjazzd" / "c.xml").exists()


def test_records_missing_files_without_raising(
    fake_xml_source: Path, tmp_path: Path
) -> None:
    out_root = tmp_path / "out"
    split = {
        "train": ["a", "missing_train"],
        "eval": ["missing_eval"],
        "test": ["e", "missing_test"],
    }

    result = route_split(fake_xml_source, split, out_root)

    assert result.n_train == 1
    assert result.n_val == 0
    assert result.n_test_canonical == 1
    assert result.missing_train == ["missing_train"]
    assert result.missing_eval == ["missing_eval"]
    assert result.missing_test == ["missing_test"]


def test_creates_output_dirs_if_absent(fake_xml_source: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "fresh"
    split = {"train": ["a"], "eval": ["b"], "test": ["c"]}

    route_split(fake_xml_source, split, out_root)

    assert (out_root / "train" / "wjazzd").is_dir()
    assert (out_root / "test" / "wjazzd").is_dir()
    assert (out_root / "test_canonical" / "wjazzd").is_dir()


def test_routes_canonical_344_43_40(tmp_path: Path) -> None:
    """Smoke against the actual SSoT split sizes."""
    src = tmp_path / "src"
    src.mkdir()
    for i in range(427):
        (src / f"f{i:03d}.xml").write_text("<dummy/>")

    train_ids = [f"f{i:03d}" for i in range(344)]
    eval_ids = [f"f{i:03d}" for i in range(344, 387)]
    test_ids = [f"f{i:03d}" for i in range(387, 427)]
    split = {"train": train_ids, "eval": eval_ids, "test": test_ids}

    result = route_split(src, split, tmp_path / "out")

    assert result.n_train == 344
    assert result.n_val == 43
    assert result.n_test_canonical == 40
    assert result.missing_train == []
    assert result.missing_eval == []
    assert result.missing_test == []
