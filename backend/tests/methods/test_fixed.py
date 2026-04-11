"""Tests for fixed-width segmentation method."""

import numpy as np

from python.acpd.methods import fixed


def test_fixed_run_basic():
    """Test fixed method returns valid result structure."""
    context = {
        "audio_length": 100,
        "num_segments": 4,
    }
    config = {}
    
    result = fixed.run(context, config)
    
    assert isinstance(result, dict)
    assert "segments" in result
    assert "suggested_labels" in result
    assert "source" in result
    assert result["source"] == "fixed-even"


def test_fixed_run_segment_count():
    """Test fixed segmentation produces exactly requested segments."""
    context = {
        "audio_length": 100,
        "num_segments": 5,
    }
    config = {}
    
    result = fixed.run(context, config)
    
    assert len(result["segments"]) == 5
    assert len(result["suggested_labels"]) == 5


def test_fixed_run_all_background():
    """Test fixed segmentation suggests all background labels."""
    context = {
        "audio_length": 50,
        "num_segments": 3,
    }
    config = {}
    
    result = fixed.run(context, config)
    
    # All segments should be labeled as background
    assert all(label == "background" for label in result["suggested_labels"])


def test_fixed_run_empty_probabilities():
    """Test fixed method returns empty probabilities (no frame-level data)."""
    context = {
        "audio_length": 100,
        "num_segments": 4,
    }
    config = {}
    
    result = fixed.run(context, config)
    
    assert result["probabilities"] == []
    assert result["frame_labels"] == []
    assert result["change_scores"] == []
