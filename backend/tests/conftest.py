"""Shared pytest fixtures and helpers for backend Python tests."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import msgpack
import numpy as np
import pytest

from python.acpd.data import load_embeddings_and_timings, to_centers


BACKEND_ROOT = Path(__file__).resolve().parents[1]
TESTS_DATA_ROOT = BACKEND_ROOT / "tests" / "data"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture
def sample_timings():
    """Return a small timing array for smoke tests."""
    return np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def sample_embeddings():
    """Return a small embedding matrix for smoke tests."""
    return np.array(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.8, 0.7, 0.6],
            [0.9, 0.8, 0.7],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def write_msgpack(tmp_path):
    """Return helper that writes a BirdNET-like msgpack fixture file."""

    def _write(relative_path, timings, embeddings):
        file_path = tmp_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timings": np.asarray(timings, dtype=np.float64).tolist(),
            "embeddings": np.asarray(embeddings, dtype=np.float64).tolist(),
        }
        file_path.write_bytes(msgpack.packb(payload, use_bin_type=True))
        return file_path

    return _write


@pytest.fixture
def sample_label_file(tmp_path):
    """Return helper that writes a simple label file."""

    def _write(relative_path="labels/example.txt"):
        file_path = tmp_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            "start_time,end_time,label\n"
            "0.0,1.0,background\n"
            "1.0,3.0,presence\n",
            encoding="utf-8",
        )
        return file_path

    return _write


@pytest.fixture
def sample_dataset(tmp_path, sample_timings, sample_embeddings, write_msgpack):
    """Create a minimal on-disk dataset layout for integration-like smoke tests."""
    labels_dir = tmp_path / "labels"
    embeddings_dir = tmp_path / "embeddings"
    labels_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    file_stem = "example"
    msgpack_path = write_msgpack(
        f"embeddings/{file_stem}.birdnet.embeddings.msgpack",
        sample_timings,
        sample_embeddings,
    )
    (labels_dir / f"{file_stem}.txt").write_text(
        "start_time,end_time,label\n"
        "0.0,1.0,background\n"
        "1.0,4.0,presence\n",
        encoding="utf-8",
    )

    return {
        "root": tmp_path,
        "labels_dir": labels_dir,
        "embeddings_dir": embeddings_dir,
        "embeddings_path": msgpack_path,
        "audio_length": 4.0,
    }


@pytest.fixture
def capture_cli(monkeypatch):
    """Return helper to run simple stdin/stdout-driven CLI functions."""

    def _capture(payload):
        stdin = io.StringIO(json.dumps(payload))
        stdout = io.StringIO()
        monkeypatch.setattr(sys, "stdin", stdin)
        monkeypatch.setattr(sys, "stdout", stdout)
        return stdout

    return _capture


@pytest.fixture
def synthetic_dataset():
    """Return helper that loads generated synthetic datasets from tests/data."""

    def _load(dataset_name):
        dataset_root = TESTS_DATA_ROOT / dataset_name
        labels_dir = dataset_root / "labels"
        embeddings_dir = dataset_root / "embeddings"
        query_path = embeddings_dir / "query.birdnet.embeddings.msgpack"

        assert dataset_root.exists(), f"Synthetic dataset not found: {dataset_root}"
        assert labels_dir.exists(), f"Missing labels directory: {labels_dir}"
        assert embeddings_dir.exists(), f"Missing embeddings directory: {embeddings_dir}"
        assert query_path.exists(), f"Missing query embeddings file: {query_path}"

        timings_arr, query_embeddings = load_embeddings_and_timings(str(query_path))
        centers = to_centers(timings_arr)
        audio_length = float(timings_arr[-1, 1]) if len(timings_arr) > 0 else 0.0

        return {
            "root": dataset_root,
            "labels_dir": str(labels_dir),
            "embeddings_dir": str(embeddings_dir),
            "timings_arr": timings_arr,
            "timings_centers": centers,
            "query_embeddings": query_embeddings,
            "audio_length": audio_length,
        }

    return _load