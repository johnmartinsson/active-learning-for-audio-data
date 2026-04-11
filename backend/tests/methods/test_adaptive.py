"""Tests for adaptive segmentation method."""

import tempfile

import numpy as np
import pytest

from python.acpd.methods import adaptive


def _default_config():
    return {
        "change_point_prominence": 0.05,
        "embedding_window_size": 2,
        "probability_window_size": 2,
    }


def _assert_segments_cover_audio(segments, audio_length):
    assert len(segments) > 0
    assert segments[0]["start"] == pytest.approx(0.0)
    assert segments[-1]["end"] == pytest.approx(float(audio_length))

    for idx in range(len(segments) - 1):
        assert segments[idx]["end"] == pytest.approx(segments[idx + 1]["start"])
        assert segments[idx]["end"] > segments[idx]["start"]

    assert segments[-1]["end"] > segments[-1]["start"]


def test_adaptive_run_fallback_to_fixed_no_embeddings():
    """Test adaptive method falls back to fixed when no embeddings available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {
            "audio_length": 100,
            "num_segments": 4,
            "query_embeddings": np.empty((0, 128)),  # No embeddings
            "timings_arr": np.array([]),
            "timings_centers": np.array([]),
            "labels_dir": tmpdir,
            "embeddings_dir": tmpdir,
            "negative_clustering_method": "kmeans",
            "num_negative_clusters": 2,
        }
        config = _default_config()
        
        result = adaptive.run(context, config)
        
        assert isinstance(result, dict)
        assert result["source"] == "fixed-fallback"
        assert len(result["segments"]) == 4
        assert result["probabilities"] == []
        assert result["frame_labels"] == []
        assert result["prototype_summary"] == {}
        _assert_segments_cover_audio(result["segments"], context["audio_length"])


def test_adaptive_run_branch_without_positive_prototypes(synthetic_dataset):
    """Only background labels should use the embedding-curve branch."""
    ds = synthetic_dataset("background_only_shift")
    context = {
        "audio_length": ds["audio_length"],
        "num_segments": 4,
        "query_embeddings": ds["query_embeddings"],
        "timings_arr": ds["timings_arr"],
        "timings_centers": ds["timings_centers"],
        "labels_dir": ds["labels_dir"],
        "embeddings_dir": ds["embeddings_dir"],
        "negative_clustering_method": "none",
        "num_negative_clusters": 1,
    }

    result = adaptive.run(context, _default_config())

    assert result["source"] == "fcpd-embedding-curve"
    assert len(result["segments"]) == 4
    assert result["probabilities"] == []
    assert result["frame_labels"] == []
    assert all(label == "background" for label in result["suggested_labels"])
    assert result["prototype_summary"]["num_positive_prototypes"] == 0
    assert result["prototype_summary"]["num_background_prototypes"] >= 1
    assert len(result["change_scores"]) == len(context["query_embeddings"])
    assert len(result["change_point_times"]) <= context["num_segments"] - 1
    _assert_segments_cover_audio(result["segments"], context["audio_length"])


def test_adaptive_run_probability_branch_with_positive_and_background(synthetic_dataset):
    """Known positive+background prototypes should use probability-curve branch."""
    ds = synthetic_dataset("binary_step")
    context = {
        "audio_length": ds["audio_length"],
        "num_segments": 4,
        "query_embeddings": ds["query_embeddings"],
        "timings_arr": ds["timings_arr"],
        "timings_centers": ds["timings_centers"],
        "labels_dir": ds["labels_dir"],
        "embeddings_dir": ds["embeddings_dir"],
        "negative_clustering_method": "none",
        "num_negative_clusters": 1,
    }

    result = adaptive.run(context, _default_config())

    assert result["source"] == "acpd-probability-curve"
    assert len(result["segments"]) == 4
    assert len(result["suggested_labels"]) == 4
    assert len(result["frame_labels"]) == len(context["query_embeddings"])
    assert len(result["change_scores"]) == len(context["query_embeddings"])
    assert len(result["change_point_times"]) <= context["num_segments"] - 1

    probabilities = np.asarray(result["probabilities"], dtype=np.float64)
    assert probabilities.ndim == 1
    assert probabilities.shape[0] == len(context["query_embeddings"])
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)

    assert result["prototype_summary"]["num_positive_prototypes"] >= 1
    assert "class_label" in result["prototype_summary"]["positive_labels"]
    assert result["prototype_summary"]["num_background_prototypes"] >= 1
    assert "class_label" in result["suggested_labels"]
    _assert_segments_cover_audio(result["segments"], context["audio_length"])


def test_adaptive_run_synthesizes_background_when_missing(synthetic_dataset):
    """If no background labels exist, method should derive background prototypes from query."""
    ds = synthetic_dataset("positive_only_shift")
    context = {
        "audio_length": ds["audio_length"],
        "num_segments": 4,
        "query_embeddings": ds["query_embeddings"],
        "timings_arr": ds["timings_arr"],
        "timings_centers": ds["timings_centers"],
        "labels_dir": ds["labels_dir"],
        "embeddings_dir": ds["embeddings_dir"],
        "negative_clustering_method": "kmeans",
        "num_negative_clusters": 2,
    }

    result = adaptive.run(context, _default_config())

    assert result["source"] == "acpd-probability-curve"
    assert result["prototype_summary"]["num_positive_prototypes"] >= 1
    assert result["prototype_summary"]["num_background_prototypes"] >= 1
    assert len(result["frame_labels"]) == len(context["query_embeddings"])
    _assert_segments_cover_audio(result["segments"], context["audio_length"])


def test_adaptive_run_is_deterministic_for_same_inputs(synthetic_dataset):
    """Running adaptive twice with same synthetic input should be stable."""
    ds = synthetic_dataset("binary_step")
    context = {
        "audio_length": ds["audio_length"],
        "num_segments": 4,
        "query_embeddings": ds["query_embeddings"],
        "timings_arr": ds["timings_arr"],
        "timings_centers": ds["timings_centers"],
        "labels_dir": ds["labels_dir"],
        "embeddings_dir": ds["embeddings_dir"],
        "negative_clustering_method": "none",
        "num_negative_clusters": 1,
    }

    first = adaptive.run(context, _default_config())
    second = adaptive.run(context, _default_config())

    assert first["source"] == second["source"]
    assert first["segments"] == second["segments"]
    assert first["suggested_labels"] == second["suggested_labels"]
    np.testing.assert_allclose(np.asarray(first["probabilities"]), np.asarray(second["probabilities"]))


def test_adaptive_run_multiclass_positive_labels(synthetic_dataset):
    """Multiclass synthetic data should expose both positive prototype labels."""
    ds = synthetic_dataset("multiclass_step")
    context = {
        "audio_length": ds["audio_length"],
        "num_segments": 4,
        "query_embeddings": ds["query_embeddings"],
        "timings_arr": ds["timings_arr"],
        "timings_centers": ds["timings_centers"],
        "labels_dir": ds["labels_dir"],
        "embeddings_dir": ds["embeddings_dir"],
        "negative_clustering_method": "none",
        "num_negative_clusters": 1,
    }

    result = adaptive.run(context, _default_config())

    assert result["source"] == "acpd-probability-curve"
    assert "class_alpha" in result["prototype_summary"]["positive_labels"]
    assert "class_beta" in result["prototype_summary"]["positive_labels"]
    assert any(label != "background" for label in result["suggested_labels"])


def test_adaptive_run_recovers_known_change_points_and_segments(tmp_path, write_msgpack):
    """Adaptive segmentation should recover a clean sequence of known transitions.

    This test constructs two prototype files (``background`` and ``class_1``)
    and a query file with six constant blocks:

    - background: frames 0-9
    - class_1: frames 10-19
    - background: frames 20-29
    - class_1: frames 30-39
    - background: frames 40-49
    - class_1: frames 50-59

    With a one-frame CPD window, the current probability-curve implementation
    produces change points at the midpoints of frames 9, 19, 29, 39, and 49,
    which map to split times 9.5, 19.5, 29.5, 39.5, and 49.5 seconds.
    """
    labels_dir = tmp_path / "labels"
    embeddings_dir = tmp_path / "embeddings"
    labels_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    frame_duration = 1.0
    frames_per_block = 10
    block_centers = [
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([5.0, 5.0, 5.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([5.0, 5.0, 5.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([5.0, 5.0, 5.0], dtype=np.float64),
    ]

    frame_count = frames_per_block * len(block_centers)
    starts = np.arange(frame_count, dtype=np.float64) * frame_duration
    timings = np.stack([starts, starts + frame_duration], axis=1)

    background_train = np.tile(block_centers[0].reshape(1, -1), (20, 1))
    class_train = np.tile(block_centers[1].reshape(1, -1), (20, 1))
    query_embeddings = np.vstack(
        [np.tile(center.reshape(1, -1), (frames_per_block, 1)) for center in block_centers]
    )

    write_msgpack("embeddings/train_background.birdnet.embeddings.msgpack", timings[:20], background_train)
    write_msgpack("embeddings/train_class_1.birdnet.embeddings.msgpack", timings[:20], class_train)
    query_path = write_msgpack("embeddings/query.birdnet.embeddings.msgpack", timings, query_embeddings)

    (labels_dir / "train_background.txt").write_text(
        "start_time,end_time,label\n0.0,20.0,background\n",
        encoding="utf-8",
    )
    (labels_dir / "train_class_1.txt").write_text(
        "start_time,end_time,label\n0.0,20.0,class_1\n",
        encoding="utf-8",
    )

    context = {
        "audio_length": float(timings[-1, 1]),
        "num_segments": 6,
        "query_embeddings": query_embeddings,
        "timings_arr": timings,
        "timings_centers": ((timings[:, 0] + timings[:, 1]) / 2.0).tolist(),
        "labels_dir": str(labels_dir),
        "embeddings_dir": str(embeddings_dir),
        "negative_clustering_method": "none",
        "num_negative_clusters": 1,
    }
    config = {
        "change_point_prominence": 0.01,
        "embedding_window_size": 1,
        "probability_window_size": 1,
    }

    result = adaptive.run(context, config)

    expected_change_points = [9.5, 19.5, 29.5, 39.5, 49.5]
    expected_segments = [
        {"start": 0.0, "end": 9.5},
        {"start": 9.5, "end": 19.5},
        {"start": 19.5, "end": 29.5},
        {"start": 29.5, "end": 39.5},
        {"start": 39.5, "end": 49.5},
        {"start": 49.5, "end": 60.0},
    ]

    assert query_path.exists()
    assert result["source"] == "acpd-probability-curve"
    assert result["change_point_times"] == pytest.approx(expected_change_points)
    assert result["segments"] == pytest.approx(expected_segments)

    probabilities = np.asarray(result["probabilities"], dtype=np.float64)
    block_means = [
        float(np.mean(probabilities[idx * frames_per_block:(idx + 1) * frames_per_block]))
        for idx in range(len(block_centers))
    ]
    assert block_means[0] < 0.1
    assert block_means[1] > 0.9
    assert block_means[2] < 0.1
    assert block_means[3] > 0.9
    assert block_means[4] < 0.1
    assert block_means[5] > 0.9

    expected_frame_labels = ["background"] * 10 + ["class_1"] * 10 + ["background"] * 10 + ["class_1"] * 10 + ["background"] * 10 + ["class_1"] * 10
    assert result["frame_labels"] == expected_frame_labels
