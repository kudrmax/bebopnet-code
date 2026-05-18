"""Tests for train.py resume helpers (state persist + epochs.csv).

We can't smoke the full training loop locally on macOS due to a
multiprocess hang in gather_data_from_xml (music21 + Python 3.12 fork).
Smoke of the full pipeline runs in Colab. These tests verify the new
resume/output primitives in isolation so the patch is covered.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from jazz_rnn.B_next_note_prediction.transformer.train import (
    EPOCHS_CSV_FILE,
    EPOCHS_CSV_HEADER,
    SUMMARY_FILE,
    TRAIN_STATE_FILE,
    _append_epoch_csv,
    _load_train_state,
    _save_train_state,
)


def test_save_load_train_state_round_trip(tmp_path: Path) -> None:
    _save_train_state(
        str(tmp_path),
        train_step=4242,
        best_val_loss=2.135,
        best_val_p_acc=0.157,
        best_val_d_acc=0.421,
        train_time_sec=1234.5,
    )
    state = _load_train_state(str(tmp_path))
    assert state == {
        "train_step": 4242,
        "best_val_loss": 2.135,
        "best_val_p_acc": 0.157,
        "best_val_d_acc": 0.421,
        "train_time_sec": 1234.5,
    }


def test_save_train_state_handles_none_best_val_loss(tmp_path: Path) -> None:
    _save_train_state(str(tmp_path), 0, None, 0.0, 0.0, 0.0)
    state = _load_train_state(str(tmp_path))
    assert state["best_val_loss"] is None


def test_load_train_state_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert _load_train_state(str(tmp_path)) is None


def test_save_train_state_overwrites_previous(tmp_path: Path) -> None:
    _save_train_state(str(tmp_path), 100, 5.0, 0.1, 0.2, 10.0)
    _save_train_state(str(tmp_path), 200, 4.0, 0.15, 0.25, 20.0)
    state = _load_train_state(str(tmp_path))
    assert state["train_step"] == 200
    assert state["best_val_loss"] == 4.0


def test_append_epoch_csv_writes_header_first_row(tmp_path: Path) -> None:
    row = {
        "step": 4000,
        "elapsed_sec": 123.45,
        "lr": 0.001,
        "val_loss": 2.5,
        "val_nll": 2.5,
        "val_p_top1": 0.13,
        "val_d_top1": 0.34,
        "val_p_entropy": 1.1,
        "val_d_entropy": 0.9,
    }
    _append_epoch_csv(str(tmp_path), row)

    csv_path = tmp_path / EPOCHS_CSV_FILE
    assert csv_path.is_file()
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    assert list(rows[0].keys()) == EPOCHS_CSV_HEADER
    assert rows[0]["step"] == "4000"
    assert rows[0]["val_loss"] == "2.5"


def test_append_epoch_csv_appends_without_duplicate_header(tmp_path: Path) -> None:
    base_row = {k: 0 for k in EPOCHS_CSV_HEADER}
    _append_epoch_csv(str(tmp_path), {**base_row, "step": 4000})
    _append_epoch_csv(str(tmp_path), {**base_row, "step": 8000})
    _append_epoch_csv(str(tmp_path), {**base_row, "step": 12000})

    csv_path = tmp_path / EPOCHS_CSV_FILE
    rows = list(csv.DictReader(csv_path.open()))
    assert [r["step"] for r in rows] == ["4000", "8000", "12000"]


def test_constants_match_documented_filenames() -> None:
    assert TRAIN_STATE_FILE == "train_state.json"
    assert EPOCHS_CSV_FILE == "epochs.csv"
    assert SUMMARY_FILE == "summary.json"
