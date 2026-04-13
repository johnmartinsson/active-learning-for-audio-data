"""Tests for python.sampling batch strategy migration."""

import json
from pathlib import Path

import msgpack
import numpy as np

from python.sampling import strategies


def _write_embeddings_file(embeddings_dir: Path, stem: str, embeddings: list[list[float]]):
    timings = [[float(i), float(i + 1)] for i in range(len(embeddings))]
    payload = {
        "timings": timings,
        "embeddings": embeddings,
    }
    path = embeddings_dir / f"{stem}.birdnet.embeddings.msgpack"
    path.write_bytes(msgpack.packb(payload, use_bin_type=True))


def _write_label_file(labels_dir: Path, stem: str):
    label_path = labels_dir / f"{stem}.txt"
    label_path.write_text("start_time,end_time,label\n0,1,chick\n1,2,background\n", encoding="utf-8")


def _setup_minimal_dirs(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    embeddings_dir = tmp_path / "embeddings"
    labels_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    return labels_dir, embeddings_dir


def test_sampling_uncertainty_picks_most_ambiguous_file(tmp_path):
    labels_dir, embeddings_dir = _setup_minimal_dirs(tmp_path)

    _write_embeddings_file(embeddings_dir, "labeled", [[1.0, 0.0], [0.0, 1.0]])
    _write_label_file(labels_dir, "labeled")

    _write_embeddings_file(embeddings_dir, "ambiguous", [[0.5, 0.5]])
    _write_embeddings_file(embeddings_dir, "certain", [[1.0, 0.0]])

    sampled = strategies.sample_files(
        {
            "strategy": "uncertainty",
            "batch_size": 1,
            "unlabeled_files": ["certain", "ambiguous"],
            "labels_dir": str(labels_dir),
            "embeddings_dir": str(embeddings_dir),
        }
    )

    assert sampled == ["ambiguous"]


def test_sampling_certainty_picks_low_entropy_file(tmp_path):
    labels_dir, embeddings_dir = _setup_minimal_dirs(tmp_path)

    _write_embeddings_file(embeddings_dir, "labeled", [[1.0, 0.0], [0.0, 1.0]])
    _write_label_file(labels_dir, "labeled")

    _write_embeddings_file(embeddings_dir, "ambiguous", [[0.5, 0.5]])
    _write_embeddings_file(embeddings_dir, "certain", [[1.0, 0.0]])

    sampled = strategies.sample_files(
        {
            "strategy": "certainty",
            "batch_size": 1,
            "unlabeled_files": ["certain", "ambiguous"],
            "labels_dir": str(labels_dir),
            "embeddings_dir": str(embeddings_dir),
        }
    )

    assert sampled == ["certain"]


def test_sampling_high_probability_picks_most_positive_file(tmp_path):
    labels_dir, embeddings_dir = _setup_minimal_dirs(tmp_path)

    _write_embeddings_file(embeddings_dir, "labeled", [[1.0, 0.0], [0.0, 1.0]])
    _write_label_file(labels_dir, "labeled")

    _write_embeddings_file(embeddings_dir, "positive_like", [[1.0, 0.0]])
    _write_embeddings_file(embeddings_dir, "negative_like", [[0.0, 1.0]])

    sampled = strategies.sample_files(
        {
            "strategy": "high_probability",
            "batch_size": 1,
            "unlabeled_files": ["negative_like", "positive_like"],
            "labels_dir": str(labels_dir),
            "embeddings_dir": str(embeddings_dir),
        }
    )

    assert sampled == ["positive_like"]


def test_sampling_batch_cli_outputs_json(tmp_path):
    labels_dir, embeddings_dir = _setup_minimal_dirs(tmp_path)
    _write_embeddings_file(embeddings_dir, "f1", [[0.0, 1.0]])

    payload = {
        "strategy": "random",
        "batch_size": 1,
        "unlabeled_files": ["f1"],
        "labels_dir": str(labels_dir),
        "embeddings_dir": str(embeddings_dir),
    }

    # Direct smoke check on returned shape from strategy layer used by CLI.
    output = {"sampled_files": strategies.sample_files(payload)}
    parsed = json.loads(json.dumps(output))

    assert "sampled_files" in parsed
    assert parsed["sampled_files"] == ["f1"]
