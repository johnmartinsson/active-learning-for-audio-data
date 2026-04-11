"""Scaffold tests for backend.python.acpd.pipeline."""

import numpy as np

from python.acpd import pipeline


def test_build_fixed_segments_smoke(xfail_not_implemented):
    segments = pipeline.build_fixed_segments(10.0, 5)
    assert len(segments) == 5
    xfail_not_implemented("pipeline.build_fixed_segments")


def test_build_segments_from_splits_smoke(xfail_not_implemented):
    segments = pipeline.build_segments_from_splits(10.0, [2.0, 5.0, 8.0], 4)
    assert len(segments) == 4
    xfail_not_implemented("pipeline.build_segments_from_splits")


def test_infer_segment_labels_smoke(xfail_not_implemented):
    segments = [{"start": 0.0, "end": 2.0}, {"start": 2.0, "end": 4.0}]
    centers = [0.5, 1.5, 2.5, 3.5]
    frame_labels = ["background", "presence", "presence", "background"]
    labels = pipeline.infer_segment_labels(segments, centers, frame_labels)
    assert isinstance(labels, list)
    xfail_not_implemented("pipeline.infer_segment_labels")


def test_infer_segment_labels_from_probabilities_smoke(xfail_not_implemented):
    segments = [{"start": 0.0, "end": 2.0}, {"start": 2.0, "end": 4.0}]
    centers = [0.5, 1.5, 2.5, 3.5]
    prototype_probabilities = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.8, 0.2],
        ],
        dtype=float,
    )
    prototype_labels = ["background", "presence"]
    labels = pipeline.infer_segment_labels_from_probabilities(
        segments,
        centers,
        prototype_probabilities,
        prototype_labels,
    )
    assert isinstance(labels, list)
    xfail_not_implemented("pipeline.infer_segment_labels_from_probabilities")


def test_build_segments_response_smoke(sample_dataset, xfail_not_implemented):
    payload = {
        "audio_length": sample_dataset["audio_length"],
        "labeling_strategy_choice": "active",
        "requested_num_segments": 3,
        "embeddings_path": str(sample_dataset["embeddings_path"]),
        "labels_dir": str(sample_dataset["labels_dir"]),
        "embeddings_dir": str(sample_dataset["embeddings_dir"]),
        "negative_clustering_method": "none",
        "num_negative_clusters": 1,
    }
    response = pipeline.build_segments_response(payload)
    assert isinstance(response, dict)
    xfail_not_implemented("pipeline.build_segments_response")