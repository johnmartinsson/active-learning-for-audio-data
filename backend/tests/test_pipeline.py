"""Unit tests for backend.python.acpd.pipeline."""

import numpy as np

from python.acpd import pipeline


def test_build_fixed_segments_properties():
    segments = pipeline.build_fixed_segments(10.0, 5)
    assert len(segments) == 5
    assert segments[0]["start"] == 0.0
    assert segments[-1]["end"] == 10.0
    assert segments[0]["end"] == segments[1]["start"]


def test_build_segments_from_splits_honors_target_count():
    segments = pipeline.build_segments_from_splits(10.0, [2.0, 5.0, 8.0], 4)
    assert len(segments) == 4
    assert segments[0]["start"] == 0.0
    assert segments[-1]["end"] == 10.0


def test_infer_segment_labels_suppresses_background_when_positive_exists():
    segments = [{"start": 0.0, "end": 2.0}, {"start": 2.0, "end": 4.0}]
    centers = [0.5, 1.5, 2.5, 3.5]
    frame_labels = ["background", "class_label", "class_label", "background"]
    labels = pipeline.infer_segment_labels(segments, centers, frame_labels)
    assert labels == ["class_label", "class_label"]


def test_infer_segment_labels_from_probabilities_applies_confidence_logic():
    segments = [{"start": 0.0, "end": 2.0}, {"start": 2.0, "end": 4.0}, {"start": 4.0, "end": 6.0}]
    centers = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    prototype_probabilities = np.array(
        [
            [0.9, 0.1],
            [0.85, 0.15],
            [0.2, 0.8],
            [0.15, 0.85],
            [0.1, 0.9],
            [0.1, 0.9],
        ],
        dtype=float,
    )
    prototype_labels = ["background", "class_label"]
    labels = pipeline.infer_segment_labels_from_probabilities(
        segments,
        centers,
        prototype_probabilities,
        prototype_labels,
    )
    assert labels == ["background", "class_label", "class_label"]


def test_build_segments_response_fixed_strategy(sample_dataset):
    payload = {
        "audio_length": sample_dataset["audio_length"],
        "labeling_strategy_choice": "fixed",
        "requested_num_segments": 3,
        "embeddings_path": str(sample_dataset["embeddings_path"]),
        "labels_dir": str(sample_dataset["labels_dir"]),
        "embeddings_dir": str(sample_dataset["embeddings_dir"]),
        "negative_clustering_method": "none",
        "num_negative_clusters": 1,
    }
    response = pipeline.build_segments_response(payload)
    assert isinstance(response, dict)
    assert len(response["segments"]) == 3
    assert len(response["suggestedLabels"]) == 3
    assert response["backend"]["source"] == "fixed-even"
    assert response["backend"]["returned_num_segments"] == 3


def test_build_segments_response_adaptive_multiclass(synthetic_dataset):
    ds = synthetic_dataset("multiclass_step")
    payload = {
        "audio_length": ds["audio_length"],
        "labeling_strategy_choice": "active",
        "adapted_method": "acpd_probability",
        "requested_num_segments": 4,
        "embeddings_path": str(ds["root"] / "embeddings" / "query.birdnet.embeddings.msgpack"),
        "labels_dir": ds["labels_dir"],
        "embeddings_dir": ds["embeddings_dir"],
        "negative_clustering_method": "none",
        "num_negative_clusters": 1,
    }
    response = pipeline.build_segments_response(payload)

    assert response["backend"]["source"] == "acpd-probability-curve"
    positive_labels = response["backend"]["prototype_summary"]["positive_labels"]
    assert "class_alpha" in positive_labels
    assert "class_beta" in positive_labels
    assert len(response["segments"]) == 4
    assert len(response["suggestedLabels"]) == 4


def test_build_segments_response_unknown_method_falls_back_to_fixed(sample_dataset):
    payload = {
        "audio_length": sample_dataset["audio_length"],
        "adapted_method": "not_a_method",
        "labeling_strategy_choice": "active",
        "requested_num_segments": 2,
        "embeddings_path": str(sample_dataset["embeddings_path"]),
        "labels_dir": str(sample_dataset["labels_dir"]),
        "embeddings_dir": str(sample_dataset["embeddings_dir"]),
        "negative_clustering_method": "none",
        "num_negative_clusters": 1,
    }

    response = pipeline.build_segments_response(payload)

    assert response["backend"]["source"] == "fixed-unknown-method"
    assert len(response["segments"]) == 2