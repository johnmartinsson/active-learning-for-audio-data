"""Generate deterministic synthetic datasets for backend tests.

This script creates small BirdNET-like msgpack/label datasets under
``backend/tests/data`` so method tests can exercise known scenarios.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import msgpack
import numpy as np


def _timings(frame_count: int, frame_duration: float) -> np.ndarray:
    starts = np.arange(frame_count, dtype=np.float64) * float(frame_duration)
    ends = starts + float(frame_duration)
    return np.stack([starts, ends], axis=1)


def _write_msgpack(path: Path, timings: np.ndarray, embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timings": np.asarray(timings, dtype=np.float64).tolist(),
        "embeddings": np.asarray(embeddings, dtype=np.float64).tolist(),
    }
    path.write_bytes(msgpack.packb(payload, use_bin_type=True))


def _write_labels(path: Path, rows: list[tuple[float, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["start_time,end_time,label"]
    lines.extend(f"{start:.6f},{end:.6f},{label}" for start, end, label in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _block_embeddings(values: list[np.ndarray], counts: list[int], noise: float = 0.0) -> np.ndarray:
    blocks = []
    rng = np.random.default_rng(seed=0)
    for center, count in zip(values, counts):
        block = np.tile(center.reshape(1, -1), (int(count), 1)).astype(np.float64)
        if noise > 0:
            block = block + rng.normal(0.0, noise, size=block.shape)
        blocks.append(block)
    return np.vstack(blocks) if blocks else np.empty((0, 0), dtype=np.float64)


def _create_binary_step(root: Path) -> None:
    """Create binary dataset with a single foreground transition in the query.

    Purpose
    -------
    Validates the adaptive probability branch when both positive and
    background prototypes are available.
    """
    dataset_root = root / "binary_step"
    labels_dir = dataset_root / "labels"
    embeddings_dir = dataset_root / "embeddings"

    frame_count = 20
    frame_duration = 0.5
    timings = _timings(frame_count, frame_duration)
    audio_length = float(timings[-1, 1])

    bg_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    fg_center = np.array([3.0, 3.0, 3.0], dtype=np.float64)

    train_bg = _block_embeddings([bg_center], [frame_count], noise=0.01)
    train_fg = _block_embeddings([fg_center], [frame_count], noise=0.01)
    query = _block_embeddings([bg_center, fg_center], [10, 10], noise=0.01)

    _write_msgpack(embeddings_dir / "train_background.birdnet.embeddings.msgpack", timings, train_bg)
    _write_msgpack(embeddings_dir / "train_class_label.birdnet.embeddings.msgpack", timings, train_fg)
    _write_msgpack(embeddings_dir / "query.birdnet.embeddings.msgpack", timings, query)

    _write_labels(labels_dir / "train_background.txt", [(0.0, audio_length, "background")])
    _write_labels(labels_dir / "train_class_label.txt", [(0.0, audio_length, "class_label")])

    metadata = {
        "name": "binary_step",
        "description": "Background then class_label in query; both prototypes available.",
        "purpose": "Exercise acpd-probability-curve branch with one known transition.",
        "expected_assertions": [
            "backend source is acpd-probability-curve",
            "at least one suggested label is class_label",
            "prototype_summary reports positive and background prototypes",
        ],
        "query_stem": "query",
        "audio_length": audio_length,
        "expected_branch": "acpd-probability-curve",
        "expected_transition_time": audio_length / 2.0,
    }
    (dataset_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _create_background_only_shift(root: Path) -> None:
    """Create dataset with only background supervision and a query shift.

    Purpose
    -------
    Validates fallback to embedding-curve CPD when no positive labels exist.
    """
    dataset_root = root / "background_only_shift"
    labels_dir = dataset_root / "labels"
    embeddings_dir = dataset_root / "embeddings"

    frame_count = 20
    frame_duration = 0.5
    timings = _timings(frame_count, frame_duration)
    audio_length = float(timings[-1, 1])

    c1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    c2 = np.array([5.0, 5.0, 5.0], dtype=np.float64)

    train_bg = _block_embeddings([c1], [frame_count], noise=0.01)
    query = _block_embeddings([c1, c2], [10, 10], noise=0.01)

    _write_msgpack(embeddings_dir / "train_background.birdnet.embeddings.msgpack", timings, train_bg)
    _write_msgpack(embeddings_dir / "query.birdnet.embeddings.msgpack", timings, query)

    _write_labels(labels_dir / "train_background.txt", [(0.0, audio_length, "background")])

    metadata = {
        "name": "background_only_shift",
        "description": "Only background labels are available; should trigger embedding-curve branch.",
        "purpose": "Exercise fcpd-embedding-curve branch with no positive prototypes.",
        "expected_assertions": [
            "backend source is fcpd-embedding-curve",
            "all suggested labels are background",
            "num_positive_prototypes is zero",
        ],
        "query_stem": "query",
        "audio_length": audio_length,
        "expected_branch": "fcpd-embedding-curve",
    }
    (dataset_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _create_positive_only_shift(root: Path) -> None:
    """Create dataset with only positive supervision and mixed query frames.

    Purpose
    -------
    Validates synthetic background prototype generation from query embeddings
    when explicit background labels are absent.
    """
    dataset_root = root / "positive_only_shift"
    labels_dir = dataset_root / "labels"
    embeddings_dir = dataset_root / "embeddings"

    frame_count = 20
    frame_duration = 0.5
    timings = _timings(frame_count, frame_duration)
    audio_length = float(timings[-1, 1])

    bg_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    fg_center = np.array([3.0, 3.0, 3.0], dtype=np.float64)

    train_fg = _block_embeddings([fg_center], [frame_count], noise=0.01)
    query = _block_embeddings([bg_center, fg_center], [10, 10], noise=0.01)

    _write_msgpack(embeddings_dir / "train_class_label.birdnet.embeddings.msgpack", timings, train_fg)
    _write_msgpack(embeddings_dir / "query.birdnet.embeddings.msgpack", timings, query)

    _write_labels(labels_dir / "train_class_label.txt", [(0.0, audio_length, "class_label")])

    metadata = {
        "name": "positive_only_shift",
        "description": "Only positive labels are available; background prototype should be synthesized.",
        "purpose": "Exercise adaptive branch where background prototypes are synthesized.",
        "expected_assertions": [
            "backend source is acpd-probability-curve",
            "num_positive_prototypes is at least one",
            "num_background_prototypes is at least one",
        ],
        "query_stem": "query",
        "audio_length": audio_length,
        "expected_branch": "acpd-probability-curve",
    }
    (dataset_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _create_multiclass_step(root: Path) -> None:
    """Create dataset with two positive classes and background.

    Purpose
    -------
    Validates multi-class prototype handling and positive label discovery in
    adaptive mode.
    """
    dataset_root = root / "multiclass_step"
    labels_dir = dataset_root / "labels"
    embeddings_dir = dataset_root / "embeddings"

    frame_count = 24
    frame_duration = 0.5
    timings = _timings(frame_count, frame_duration)
    audio_length = float(timings[-1, 1])

    bg_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    c1_center = np.array([3.0, 3.0, 0.0], dtype=np.float64)
    c2_center = np.array([0.0, 3.0, 3.0], dtype=np.float64)

    train_bg = _block_embeddings([bg_center], [frame_count], noise=0.01)
    train_c1 = _block_embeddings([c1_center], [frame_count], noise=0.01)
    train_c2 = _block_embeddings([c2_center], [frame_count], noise=0.01)
    query = _block_embeddings([bg_center, c1_center, c2_center], [8, 8, 8], noise=0.01)

    _write_msgpack(embeddings_dir / "train_background.birdnet.embeddings.msgpack", timings, train_bg)
    _write_msgpack(embeddings_dir / "train_class_alpha.birdnet.embeddings.msgpack", timings, train_c1)
    _write_msgpack(embeddings_dir / "train_class_beta.birdnet.embeddings.msgpack", timings, train_c2)
    _write_msgpack(embeddings_dir / "query.birdnet.embeddings.msgpack", timings, query)

    _write_labels(labels_dir / "train_background.txt", [(0.0, audio_length, "background")])
    _write_labels(labels_dir / "train_class_alpha.txt", [(0.0, audio_length, "class_alpha")])
    _write_labels(labels_dir / "train_class_beta.txt", [(0.0, audio_length, "class_beta")])

    metadata = {
        "name": "multiclass_step",
        "description": "Background followed by class_alpha then class_beta in query.",
        "purpose": "Exercise multiclass positive prototype handling in adaptive mode.",
        "expected_assertions": [
            "positive_labels contain class_alpha and class_beta",
            "backend source is acpd-probability-curve",
            "at least one non-background suggested label is returned",
        ],
        "query_stem": "query",
        "audio_length": audio_length,
        "expected_branch": "acpd-probability-curve",
    }
    (dataset_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic datasets for tests.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "tests" / "data"),
        help="Directory where synthetic datasets are written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _create_binary_step(output_dir)
    _create_background_only_shift(output_dir)
    _create_positive_only_shift(output_dir)
    _create_multiclass_step(output_dir)

    summary = {
        "output_dir": str(output_dir),
        "datasets": [
            "binary_step",
            "background_only_shift",
            "positive_only_shift",
            "multiclass_step",
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
