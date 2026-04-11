"""Tests for adaptive segmentation method."""

import tempfile
import numpy as np

from python.acpd.methods import adaptive


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
        config = {
            "change_point_prominence": 0.1,
            "embedding_window_size": 10,
            "probability_window_size": 10,
        }
        
        result = adaptive.run(context, config)
        
        assert isinstance(result, dict)
        assert result["source"] == "fixed-fallback"
        assert len(result["segments"]) == 4


def test_adaptive_run_returns_valid_structure(sample_embeddings):
    """Test adaptive method returns expected result structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {
            "audio_length": 100,
            "num_segments": 4,
            "query_embeddings": sample_embeddings[:4],
            "timings_arr": np.linspace(0, 100, 4),
            "timings_centers": np.linspace(1, 99, 4),
            "labels_dir": tmpdir,
            "embeddings_dir": tmpdir,
            "negative_clustering_method": "none",
            "num_negative_clusters": 1,
        }
        config = {
            "change_point_prominence": 0.1,
            "embedding_window_size": 5,
            "probability_window_size": 5,
        }
        
        result = adaptive.run(context, config)
        
        assert "segments" in result
        assert "suggested_labels" in result
        assert "source" in result
        assert "probabilities" in result
        assert "frame_labels" in result


def test_adaptive_run_segment_count_reasonable(sample_embeddings):
    """Test adaptive segmentation respects approximate num_segments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {
            "audio_length": 200,
            "num_segments": 5,
            "query_embeddings": sample_embeddings[:4],
            "timings_arr": np.linspace(0, 200, 4),
            "timings_centers": np.linspace(1, 199, 4),
            "labels_dir": tmpdir,
            "embeddings_dir": tmpdir,
            "negative_clustering_method": "none",
            "num_negative_clusters": 1,
        }
        config = {
            "change_point_prominence": 0.05,
            "embedding_window_size": 5,
            "probability_window_size": 5,
        }
        
        result = adaptive.run(context, config)
        
        # Should produce reasonable number of segments (not exact due to CPD)
        assert len(result["segments"]) > 0
        assert len(result["segments"]) <= max(context["num_segments"] * 2, 10)


def test_adaptive_run_labels_match_segments(sample_embeddings):
    """Test adaptive suggested labels match segment count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {
            "audio_length": 100,
            "num_segments": 4,
            "query_embeddings": sample_embeddings[:4],
            "timings_arr": np.linspace(0, 100, 4),
            "timings_centers": np.linspace(1, 99, 4),
            "labels_dir": tmpdir,
            "embeddings_dir": tmpdir,
            "negative_clustering_method": "none",
            "num_negative_clusters": 1,
        }
        config = {
            "change_point_prominence": 0.1,
            "embedding_window_size": 5,
            "probability_window_size": 5,
        }
        
        result = adaptive.run(context, config)
        
        # Number of labels should match number of segments
        assert len(result["suggested_labels"]) == len(result["segments"])
