"""Route WjazzD xml files into BebopNet's directory layout.

Reads diploma2/pipelines/training-pipeline/wjazzd_split.json (canonical
cross-model split: train=344, eval=43, test=40) and copies xml files from
a single source pool into three buckets:

    split.json["train"] (344) -> out_root/train/wjazzd/          (-> train.pkl,
                                                                  gradient steps)
    split.json["eval"]  (43)  -> out_root/test/wjazzd/           (-> val.pkl,
                                                                  best-checkpoint
                                                                  selection)
    split.json["test"]  (40)  -> out_root/test_canonical/wjazzd/ (held out from
                                                                  both training
                                                                  and selection;
                                                                  used only for
                                                                  the post-train
                                                                  evaluate step
                                                                  in evaluate_canonical.py)

Authorial ``gather_data_from_xml.py:76-77`` globs ``train/*/*.xml`` and
``test/*/*.xml`` (two-level). The third bucket lives at ``test_canonical/``,
outside those globs, so the canonical 40 are not pulled into train.pkl/val.pkl.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class RoutingResult:
    train_routed: List[str]
    val_routed: List[str]
    test_canonical_routed: List[str]
    missing_train: List[str]
    missing_eval: List[str]
    missing_test: List[str]

    @property
    def n_train(self) -> int:
        return len(self.train_routed)

    @property
    def n_val(self) -> int:
        return len(self.val_routed)

    @property
    def n_test_canonical(self) -> int:
        return len(self.test_canonical_routed)


def route_split(
    xml_source: Path,
    split: Dict[str, List[str]],
    out_root: Path,
) -> RoutingResult:
    """Copy xml from ``xml_source`` into 3 buckets per :mod:`module docstring`.

    Files missing from ``xml_source`` are recorded in the result but do not
    raise — physical reality may differ from split.json (e.g., MINGUS xml
    converter fails on some solos).
    """
    train_dir = out_root / "train" / "wjazzd"
    val_dir = out_root / "test" / "wjazzd"  # author's "test" path -> val.pkl
    test_canonical_dir = out_root / "test_canonical" / "wjazzd"
    for d in (train_dir, val_dir, test_canonical_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_routed: List[str] = []
    val_routed: List[str] = []
    test_canonical_routed: List[str] = []
    missing_train: List[str] = []
    missing_eval: List[str] = []
    missing_test: List[str] = []

    routes = (
        ("train", train_dir, train_routed, missing_train),
        ("eval", val_dir, val_routed, missing_eval),
        ("test", test_canonical_dir, test_canonical_routed, missing_test),
    )
    for bucket, dest_dir, routed, missing in routes:
        for song_id in split.get(bucket, []):
            src = xml_source / f"{song_id}.xml"
            if src.is_file():
                shutil.copy2(src, dest_dir / src.name)
                routed.append(song_id)
            else:
                missing.append(song_id)

    return RoutingResult(
        train_routed=sorted(train_routed),
        val_routed=sorted(val_routed),
        test_canonical_routed=sorted(test_canonical_routed),
        missing_train=sorted(missing_train),
        missing_eval=sorted(missing_eval),
        missing_test=sorted(missing_test),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--xml-source",
        type=Path,
        required=True,
        help="Source pool of xml files (one flat dir, NNN_*.xml)",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        required=True,
        help="Path to wjazzd_split.json (cross-model SSoT)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root; populates {train,test,test_canonical}/wjazzd/ underneath",
    )
    args = parser.parse_args()

    if not args.xml_source.is_dir():
        raise FileNotFoundError(f"xml-source not found: {args.xml_source}")
    if not args.split_json.is_file():
        raise FileNotFoundError(f"split-json not found: {args.split_json}")

    split = json.loads(args.split_json.read_text())
    result = route_split(args.xml_source, split, args.out_root)

    print(
        f"Routed: train={result.n_train} (target {len(split['train'])}), "
        f"val={result.n_val} (target {len(split['eval'])}), "
        f"test_canonical={result.n_test_canonical} (target {len(split['test'])})"
    )
    if result.missing_train or result.missing_eval or result.missing_test:
        print(
            f"Missing from source: train={len(result.missing_train)}, "
            f"eval={len(result.missing_eval)}, test={len(result.missing_test)}"
        )
        for bucket, songs in (
            ("train", result.missing_train),
            ("eval", result.missing_eval),
            ("test", result.missing_test),
        ):
            for song in songs:
                print(f"  [{bucket}] {song}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
