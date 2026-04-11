"""Unit tests for backend.python.acpd.change_detection."""

import numpy as np

from python.acpd import change_detection


def test_cosine_distance_score_expected_values():
    assert change_detection.cosine_distance_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0
    assert change_detection.cosine_distance_score(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0
    assert change_detection.cosine_distance_score(np.array([0.0, 0.0]), np.array([0.0, 1.0])) == 0.0


def test_euclidean_distance_score_expected_values():
    assert change_detection.euclidean_distance_score(np.array([0.0]), np.array([1.0])) == 1.0
    assert change_detection.euclidean_distance_score(np.array([1.0, 2.0]), np.array([4.0, 6.0])) == 5.0


def test_distance_past_and_future_averages_step_signal():
    sequence = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    scores = change_detection.distance_past_and_future_averages(
        sequence,
        change_detection.euclidean_distance_score,
        offset=0,
        window_size=1,
    )
    assert len(scores) == len(sequence)
    assert scores[0] == 0.0
    assert scores[-1] == 0.0
    assert float(np.max(scores)) > 0.0


def test_rank_change_point_peaks_returns_sorted_indices():
    peak_indices, prominences = change_detection.rank_change_point_peaks([0.0, 1.0, 0.0, 2.0, 0.0], 0.0, 2)
    assert peak_indices == sorted(peak_indices)
    assert len(peak_indices) <= 2
    assert len(prominences) <= 2


def test_peak_times_from_indices(sample_timings):
    peak_times = change_detection.peak_times_from_indices(sample_timings, [1, 2])
    assert peak_times == [1.5, 2.5]


def test_detect_change_points_from_probabilities_shapes(sample_timings):
    peak_times, scores, peak_indices, prominences = change_detection.detect_change_points_from_probabilities(
        [0.0, 0.0, 1.0, 1.0],
        sample_timings,
        n_peaks=2,
        prominence=0.0,
        window_size=1,
    )
    assert len(scores) == len(sample_timings)
    assert len(peak_indices) <= 2
    assert len(peak_times) == len(peak_indices)
    assert len(prominences) <= 2


def test_detect_change_points_from_embeddings_shapes(sample_embeddings, sample_timings):
    peak_times, scores, peak_indices, prominences = change_detection.detect_change_points_from_embeddings(
        sample_embeddings,
        sample_timings,
        n_peaks=2,
        prominence=0.0,
        window_size=1,
    )
    assert len(scores) == len(sample_embeddings)
    assert len(peak_indices) <= 2
    assert len(peak_times) == len(peak_indices)
    assert len(prominences) <= 2