"""Integration tests for backend.python.acpd.get_segments_cli."""

import json

from python.acpd import get_segments_cli


def test_main_fixed_strategy(sample_dataset, capture_cli):
    payload = {
        "audio_length": sample_dataset["audio_length"],
        "labeling_strategy_choice": "fixed",
        "requested_num_segments": 2,
        "embeddings_path": str(sample_dataset["embeddings_path"]),
        "labels_dir": str(sample_dataset["labels_dir"]),
        "embeddings_dir": str(sample_dataset["embeddings_dir"]),
        "negative_clustering_method": "none",
        "num_negative_clusters": 1,
    }
    stdout = capture_cli(payload)
    get_segments_cli.main()
    response = json.loads(stdout.getvalue())
    assert isinstance(response, dict)
    assert len(response["segments"]) == 2
    assert response["backend"]["source"] == "fixed-even"


def test_main_adaptive_strategy_with_synthetic_dataset(synthetic_dataset, capture_cli):
    ds = synthetic_dataset("binary_step")
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

    stdout = capture_cli(payload)
    get_segments_cli.main()
    response = json.loads(stdout.getvalue())

    assert isinstance(response, dict)
    assert response["backend"]["source"] == "acpd-probability-curve"
    assert len(response["segments"]) == 4
    assert "class_label" in response["backend"]["prototype_summary"]["positive_labels"]