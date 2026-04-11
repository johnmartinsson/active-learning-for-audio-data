"""Scaffold tests for backend.python.acpd.get_segments_cli."""

import json

from python.acpd import get_segments_cli


def test_main_smoke(sample_dataset, capture_cli, xfail_not_implemented):
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
    xfail_not_implemented("get_segments_cli.main")