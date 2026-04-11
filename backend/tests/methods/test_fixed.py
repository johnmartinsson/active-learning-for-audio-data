"""Tests for fixed-width segmentation method."""

import pytest

from python.acpd.methods import fixed


def _assert_segments_cover_audio(segments, audio_length):
    assert len(segments) > 0
    assert segments[0]["start"] == pytest.approx(0.0)
    assert segments[-1]["end"] == pytest.approx(float(audio_length))

    for idx in range(len(segments) - 1):
        assert segments[idx]["end"] == pytest.approx(segments[idx + 1]["start"])
        assert segments[idx]["end"] > segments[idx]["start"]

    assert segments[-1]["end"] > segments[-1]["start"]


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
    _assert_segments_cover_audio(result["segments"], context["audio_length"])


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
    _assert_segments_cover_audio(result["segments"], context["audio_length"])


def test_fixed_run_all_background():
    """Test fixed segmentation suggests all background labels."""
    context = {
        "audio_length": 50,
        "num_segments": 3,
    }
    config = {}
    
    result = fixed.run(context, config)
    
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


def test_fixed_run_single_segment_bounds():
    """Single segment should span the complete audio interval."""
    context = {
        "audio_length": 12.5,
        "num_segments": 1,
    }

    result = fixed.run(context, config={})

    assert result["segments"] == [{"start": 0.0, "end": 12.5}]
    assert result["suggested_labels"] == ["background"]


def test_fixed_run_many_segments_remain_ordered():
    """Segments stay contiguous and ordered with larger segment budgets."""
    context = {
        "audio_length": 10.0,
        "num_segments": 25,
    }

    result = fixed.run(context, config={})

    assert len(result["segments"]) == 25
    _assert_segments_cover_audio(result["segments"], context["audio_length"])


def test_fixed_run_invalid_num_segments_raises():
    """Current behavior for non-positive segment counts should raise."""
    context = {
        "audio_length": 10.0,
        "num_segments": 0,
    }

    with pytest.raises(ZeroDivisionError):
        fixed.run(context, config={})
