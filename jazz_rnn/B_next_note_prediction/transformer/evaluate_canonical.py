"""Evaluate model_best.pt on the canonical test=40 set.

Held-out from training (gradient steps) AND from selection (val_loss best
checkpoint). After train.py finishes, this script:

  1. gather'ит 40 test xml in ``--test-canonical-xml-dir`` (=
     ``<prep_root>/test_canonical/wjazzd/``) using the converter cached at
     ``<work_dir>/converter_and_duration.pkl`` so vocabulary matches.
  2. loads ``<work_dir>/model_best.pt`` (selected on split.json[eval]=43
     during training).
  3. runs the same evaluate loop as Trainer.evaluate.
  4. appends ``final_test_loss``, ``final_test_ppl``, ``final_test_metrics``,
     ``final_test_n_files`` to ``<work_dir>/summary.json``.

Notes:
  - The author's gather_data_from_xml.py expects ``train/*/*.xml`` +
    ``test/*/*.xml`` two-level globs. We hand it an ad-hoc layout where
    test_canonical xml are placed under ``<tmp>/test/wjazzd/`` so the test
    bucket is picked up; the train bucket stays empty (its train.pkl is
    discarded).
  - This script does NOT touch model.pt / model_best.pt / training state.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from jazz_rnn.B_next_note_prediction.transformer.data_utils import (
    LMOrderedIterator,
    transpose_data_torch,
)
from jazz_rnn.B_next_note_prediction.transformer.mem_transformer import MemTransformerLM


METERS = [
    "loss", "nll", "p_nll", "d_nll",
    "p_top1", "p_top3", "p_top5", "d_top1", "d_top3",
    "p_entropy", "d_entropy", "t_entropy",
]


def _gather_test_canonical(
    test_xml_dir: Path,
    out_pkl_dir: Path,
    cached_converter: Path,
    num_processes: int,
) -> None:
    """Run author's gather_data_from_xml on canonical 40 in an ad-hoc layout."""
    out_pkl_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as ad_hoc_root:
        ad_hoc = Path(ad_hoc_root)
        # gather globs train/*/*.xml + test/*/*.xml; we want the canonical 40
        # to be picked up as "test_songs", so place them under test/wjazzd/.
        ad_hoc_test = ad_hoc / "test" / "wjazzd"
        (ad_hoc / "train" / "wjazzd").mkdir(parents=True, exist_ok=True)
        ad_hoc_test.mkdir(parents=True, exist_ok=True)
        for xml in test_xml_dir.glob("*.xml"):
            shutil.copy2(xml, ad_hoc_test / xml.name)

        cmd = [
            sys.executable, "-m", "jazz_rnn.A_data_prep.gather_data_from_xml",
            "--xml_dir", str(ad_hoc),
            "--out_dir", str(out_pkl_dir),
            "--cached_converter", str(cached_converter),
            "--num_processes", str(num_processes),
        ]
        print(">>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def _load_test_iter(
    pkl_dir: Path, eval_tgt_len: int, ext_len: int, batch_size: int, device: torch.device,
) -> tuple:
    with open(pkl_dir / "converter_and_duration.pkl", "rb") as f:
        converter = pickle.load(f)
    with open(pkl_dir / "val.pkl", "rb") as f:
        test_data = pickle.load(f)

    flat = np.concatenate([v for v in test_data.values()], axis=0)
    test_torch = torch.from_numpy(flat)
    transposed = [transpose_data_torch(i, test_torch) for i in range(-5, 7)]
    test_torch = torch.cat(transposed)

    test_iter = LMOrderedIterator(
        test_torch, batch_size, eval_tgt_len, device=device, ext_len=ext_len,
    )
    return test_iter, converter, sum(1 for _ in test_data)


def _build_model(work_dir: Path, converter, device: torch.device) -> MemTransformerLM:
    with open(work_dir / "args.json") as f:
        kwargs = json.load(f)
    # converter is dropped from args.json on save; reattach.
    kwargs["converter"] = converter
    # JSON deserializes lists; some kwargs are tuple-typed. Restore.
    for k in ("pitch_sizes", "duration_sizes", "offset_sizes"):
        if k in kwargs and isinstance(kwargs[k], list):
            kwargs[k] = tuple(kwargs[k])

    model = MemTransformerLM(**kwargs)
    state = torch.load(work_dir / "model_best.pt", map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def _evaluate(model: MemTransformerLM, test_iter, args_dict: dict) -> tuple[float, dict]:
    tgt_len = int(args_dict["tgt_len"])
    eval_tgt_len = int(args_dict["eval_tgt_len"])
    ext_len = int(args_dict["ext_len"])
    mem_len = int(args_dict["mem_len"])

    if mem_len == 0:
        model.reset_length(eval_tgt_len, ext_len + tgt_len - eval_tgt_len, mem_len)
    else:
        model.reset_length(eval_tgt_len, ext_len, mem_len + tgt_len - eval_tgt_len)

    totals = {k: 0.0 for k in METERS}
    total_loss = 0.0
    total_len = 0
    with torch.no_grad():
        mems = tuple()
        for data, target, seq_len in test_iter:
            prediction, ret, loss_dict, _ = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.float().mean().type_as(loss)
            for k, v in loss_dict.items():
                totals[k] += seq_len * float(v)
            total_loss += seq_len * float(loss.item())
            total_len += seq_len

    avg_loss = total_loss / max(total_len, 1)
    avg_metrics = {k: v / max(total_len, 1) for k, v in totals.items()}
    return avg_loss, avg_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="Train work_dir with model_best.pt + args.json + summary.json")
    parser.add_argument("--test-canonical-xml-dir", type=Path, required=True,
                        help="Directory with the canonical 40 test xml files")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    if not (args.work_dir / "model_best.pt").is_file():
        raise FileNotFoundError(f"model_best.pt missing under {args.work_dir}")
    if not (args.work_dir / "args.json").is_file():
        raise FileNotFoundError(f"args.json missing under {args.work_dir}")
    if not (args.work_dir / "summary.json").is_file():
        raise FileNotFoundError(f"summary.json missing under {args.work_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"==> Device: {device}", flush=True)

    with tempfile.TemporaryDirectory() as test_pkl_root:
        test_pkl_dir = Path(test_pkl_root)
        _gather_test_canonical(
            test_xml_dir=args.test_canonical_xml_dir,
            out_pkl_dir=test_pkl_dir,
            cached_converter=args.work_dir / "converter_and_duration.pkl",
            num_processes=args.num_processes,
        )

        with open(args.work_dir / "args.json") as f:
            saved_args = json.load(f)
        # tgt_len etc. may be needed at runtime; pull from summary.config
        # which was saved by train.py main().
        with open(args.work_dir / "summary.json") as f:
            summary = json.load(f)
        run_args = summary.get("config", {})
        for k in ("tgt_len", "eval_tgt_len", "ext_len", "mem_len", "batch_size"):
            saved_args.setdefault(k, run_args.get(k, 64))

        test_iter, converter, n_files = _load_test_iter(
            test_pkl_dir,
            eval_tgt_len=int(saved_args.get("eval_tgt_len", 64)),
            ext_len=int(saved_args.get("ext_len", 0)),
            batch_size=int(saved_args.get("batch_size", 32)),
            device=device,
        )
        model = _build_model(args.work_dir, converter, device)
        avg_loss, avg_metrics = _evaluate(model, test_iter, saved_args)

    summary["final_test_loss"] = float(avg_loss)
    summary["final_test_ppl"] = float(math.exp(avg_loss))
    summary["final_test_metrics"] = {k: float(v) for k, v in avg_metrics.items()}
    summary["final_test_n_files"] = int(n_files)
    summary["final_test_note"] = (
        "evaluated on canonical test=40 (split.json[test]); held out from "
        "both gradient training and best-checkpoint selection"
    )
    with open(args.work_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(
        f"==> final_test_ppl={summary['final_test_ppl']:.3f} on {n_files} files",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
