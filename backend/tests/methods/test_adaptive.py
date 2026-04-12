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

    With a one-frame CPD window, boundary-based probability CPD places change
    points on frame boundaries at 10, 20, 30, 40, and 50 seconds.
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

    expected_change_points = [10.0, 20.0, 30.0, 40.0, 50.0]
    expected_segments = [
        {"start": 0.0, "end": 10.0},
        {"start": 10.0, "end": 20.0},
        {"start": 20.0, "end": 30.0},
        {"start": 30.0, "end": 40.0},
        {"start": 40.0, "end": 50.0},
        {"start": 50.0, "end": 60.0},
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


def test_adaptive_run_detects_change_points_near_signal_edges(tmp_path, write_msgpack):
    """Detect two transitions located close to start and end of the query.

    Query layout (30 frames, 1 second each):
    - background: frames 0-2
    - class_label: frames 3-26
    - background: frames 27-29

    With window size 1, expected split times are at 3.0 and 27.0 seconds,
    which are intentionally near the signal boundaries.
    """
    labels_dir = tmp_path / "labels"
    embeddings_dir = tmp_path / "embeddings"
    labels_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    frame_duration = 1.0
    frame_count = 30
    starts = np.arange(frame_count, dtype=np.float64) * frame_duration
    timings = np.stack([starts, starts + frame_duration], axis=1)

    background_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    class_center = np.array([4.0, 4.0, 4.0], dtype=np.float64)

    background_train = np.tile(background_center.reshape(1, -1), (20, 1))
    class_train = np.tile(class_center.reshape(1, -1), (20, 1))

    query_embeddings = np.vstack(
        [
            np.tile(background_center.reshape(1, -1), (3, 1)),
            np.tile(class_center.reshape(1, -1), (24, 1)),
            np.tile(background_center.reshape(1, -1), (3, 1)),
        ]
    )

    write_msgpack("embeddings/train_background.birdnet.embeddings.msgpack", timings[:20], background_train)
    write_msgpack("embeddings/train_class_label.birdnet.embeddings.msgpack", timings[:20], class_train)
    write_msgpack("embeddings/query.birdnet.embeddings.msgpack", timings, query_embeddings)

    (labels_dir / "train_background.txt").write_text(
        "start_time,end_time,label\n0.0,20.0,background\n",
        encoding="utf-8",
    )
    (labels_dir / "train_class_label.txt").write_text(
        "start_time,end_time,label\n0.0,20.0,class_label\n",
        encoding="utf-8",
    )

    context = {
        "audio_length": float(timings[-1, 1]),
        "num_segments": 3,
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

    expected_change_points = [3.0, 27.0]
    expected_segments = [
        {"start": 0.0, "end": 3.0},
        {"start": 3.0, "end": 27.0},
        {"start": 27.0, "end": 30.0},
    ]

    assert result["source"] == "acpd-probability-curve"
    assert len(result["segments"]) == 3
    assert result["change_point_times"] == pytest.approx(expected_change_points)
    assert result["segments"] == pytest.approx(expected_segments)

    probabilities = np.asarray(result["probabilities"], dtype=np.float64)
    assert np.mean(probabilities[:3]) < 0.1
    assert np.mean(probabilities[3:27]) > 0.9
    assert np.mean(probabilities[27:]) < 0.1


def test_adaptive_run_brackets_single_frame_events(tmp_path, write_msgpack):
    """Each single-frame foreground event must produce exactly two change points:
    one to its LEFT (before event start) and one to its RIGHT (after event end).

    Query layout (17 frames, 1 second each):
        bg × 5  →  class_label × 1  →  bg × 5  →  class_label × 1  →  bg × 5

    With window_size=1 boundary scoring compares adjacent frames and assigns the
    result to the right-frame boundary index. Expected event brackets are on the
    exact boundaries:
        - event 1 LEFT  : 5.0
        - event 1 RIGHT : 6.0
        - event 2 LEFT  : 11.0
        - event 2 RIGHT : 12.0

    The test asserts the structural property: for each event there exists exactly
    one change point strictly before event_start and one strictly after event_end.
    """
    labels_dir = tmp_path / "labels"
    embeddings_dir = tmp_path / "embeddings"
    labels_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    frame_duration = 1.0
    # Layout indices: bg=0..4, event1=5, bg=6..10, event2=11, bg=12..16
    background_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    class_center = np.array([5.0, 5.0, 5.0], dtype=np.float64)

    frame_count = 17
    starts = np.arange(frame_count, dtype=np.float64) * frame_duration
    timings = np.stack([starts, starts + frame_duration], axis=1)

    event1_start = timings[5, 0]   # 5.0
    event1_end   = timings[5, 1]   # 6.0
    event2_start = timings[11, 0]  # 11.0
    event2_end   = timings[11, 1]  # 12.0

    query_embeddings = np.vstack([
        np.tile(background_center, (5, 1)),
        class_center.reshape(1, -1),          # single-frame event 1
        np.tile(background_center, (5, 1)),
        class_center.reshape(1, -1),          # single-frame event 2
        np.tile(background_center, (5, 1)),
    ])

    background_train = np.tile(background_center, (20, 1))
    class_train = np.tile(class_center, (20, 1))
    train_timings = np.stack([np.arange(20.0), np.arange(1.0, 21.0)], axis=1)

    write_msgpack("embeddings/train_background.birdnet.embeddings.msgpack", train_timings, background_train)
    write_msgpack("embeddings/train_class_label.birdnet.embeddings.msgpack", train_timings, class_train)
    write_msgpack("embeddings/query.birdnet.embeddings.msgpack", timings, query_embeddings)

    (labels_dir / "train_background.txt").write_text(
        "start_time,end_time,label\n0.0,20.0,background\n", encoding="utf-8"
    )
    (labels_dir / "train_class_label.txt").write_text(
        "start_time,end_time,label\n0.0,20.0,class_label\n", encoding="utf-8"
    )

    context = {
        "audio_length": float(timings[-1, 1]),
        "num_segments": 5,
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

    assert result["source"] == "acpd-probability-curve"

    cps = sorted(result["change_point_times"])
    # Must detect at least 4 change points (2 per event)
    assert len(cps) >= 4, f"Expected ≥4 change points, got {len(cps)}: {cps}"

    # Identify the two change points flanking event 1
    left_of_event1 = [t for t in cps if t <= event1_start]
    right_of_event1 = [t for t in cps if t >= event1_end]
    assert len(left_of_event1) >= 1, (
        f"No change point LEFT of event1 [{event1_start}, {event1_end}); cps={cps}"
    )
    assert len(right_of_event1) >= 1, (
        f"No change point RIGHT of event1 [{event1_start}, {event1_end}); cps={cps}"
    )

    # Identify the two change points flanking event 2
    left_of_event2 = [t for t in cps if t <= event2_start]
    right_of_event2 = [t for t in cps if t >= event2_end]
    assert len(left_of_event2) >= 1, (
        f"No change point LEFT of event2 [{event2_start}, {event2_end}); cps={cps}"
    )
    assert len(right_of_event2) >= 1, (
        f"No change point RIGHT of event2 [{event2_start}, {event2_end}); cps={cps}"
    )

    # The closest right-of-event1 CP must still be LEFT of event2
    assert min(right_of_event1) < event2_start, (
        f"Right-of-event1 CP should precede event2; cps={cps}"
    )

    # Probability sanity checks
    probs = np.asarray(result["probabilities"], dtype=np.float64)
    assert probs[5] > 0.9,  f"Event1 frame probability should be high, got {probs[5]:.3f}"
    assert probs[11] > 0.9, f"Event2 frame probability should be high, got {probs[11]:.3f}"
    assert np.mean(probs[[0, 1, 2, 3, 4]]) < 0.1, "Background frames before event1 should be low"
    assert np.mean(probs[[6, 7, 8, 9, 10]]) < 0.1, "Background frames between events should be low"
    assert np.mean(probs[[12, 13, 14, 15, 16]]) < 0.1, "Background frames after event2 should be low"
