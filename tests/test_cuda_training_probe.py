"""Regression checks for phase timing, measurements and runtime projection."""

import csv
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_cuda_fno_training as P


def test_projection_counts_validation_and_final_checkpoint():
    estimate = P.projection(60, 6, 3, 60, 12, 2, 10)
    assert estimate["hours_per_run"] == pytest.approx((3600 + 180 + 21) / 3600)
    assert estimate["hours_for_study"] == pytest.approx(12 * estimate["hours_per_run"])
    assert estimate["planning_hours_with_25_percent_margin"] == pytest.approx(15 * estimate["hours_per_run"])


def test_memory_values_are_in_mib(monkeypatch):
    for method in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, method, lambda: 2**20)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (3 * 2**20, 4 * 2**20))
    assert P.memory() == dict(allocated_mib=1, reserved_mib=1, peak_allocated_mib=1,
                              peak_reserved_mib=1, free_mib=3)


def test_csv_records_one_row_per_completed_update(tmp_path):
    probe = P.Probe(tmp_path)
    for update in range(2):
        probe.begin(0)
        probe.finish(torch.tensor(float(update)))
    with (tmp_path / "training_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["update"] for row in rows] == ["1", "2"]
    assert all(float(row["update_seconds"]) >= 0 for row in rows)


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_nonfinite_training_loss_is_reported(tmp_path, value):
    probe = P.Probe(tmp_path)
    probe.begin(0)
    with pytest.raises(RuntimeError, match="Non-finite training loss"):
        probe.finish(torch.tensor(value))
