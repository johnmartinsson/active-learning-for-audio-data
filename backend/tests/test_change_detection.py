"""Scaffold tests for backend.python.acpd.change_detection."""

import numpy as np

from python.acpd import change_detection


def test_cosine_distance_score_smoke(xfail_not_implemented):
    distance = change_detection.cosine_distance_score(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert isinstance(distance, float)
    xfail_not_implemented("change_detection.cosine_distance_score")


def test_euclidean_distance_score_smoke(xfail_not_implemented):
    distance = change_detection.euclidean_distance_score(np.array([0.0]), np.array([1.0]))
    assert isinstance(distance, float)
    xfail_not_implemented("change_detection.euclidean_distance_score")


def test_distance_past_and_future_averages_smoke(sample_embeddings, xfail_not_implemented):
    scores = change_detection.distance_past_and_future_averages(
        sample_embeddings,
        change_detection.euclidean_distance_score,
        offset=0,
        window_size=1,
    )
    assert len(scores) == len(sample_embeddings)
    xfail_not_implemented("change_detection.distance_past_and_future_averages")


def test_rank_change_point_peaks_smoke(xfail_not_implemented):
    peak_indices, prominences = change_detection.rank_change_point_peaks([0.0, 1.0, 0.0, 2.0, 0.0], 0.0, 2)
    assert isinstance(peak_indices, list)
    assert isinstance(prominences, list)
    xfail_not_implemented("change_detection.rank_change_point_peaks")


def test_peak_times_from_indices_smoke(sample_timings, xfail_not_implemented):
    peak_times = change_detection.peak_times_from_indices(sample_timings, [1, 2])
    assert isinstance(peak_times, list)
    xfail_not_implemented("change_detection.peak_times_from_indices")


def test_detect_change_points_from_probabilities_smoke(sample_timings, xfail_not_implemented):
    result = change_detection.detect_change_points_from_probabilities(
        [0.1, 0.9, 0.2, 0.8],
        sample_timings,
        n_peaks=2,
    )
    assert len(result) == 4
    xfail_not_implemented("change_detection.detect_change_points_from_probabilities")


def test_detect_change_points_from_embeddings_smoke(sample_embeddings, sample_timings, xfail_not_implemented):
    result = change_detection.detect_change_points_from_embeddings(
        sample_embeddings,
        sample_timings,
        n_peaks=2,
    )
    assert len(result) == 4
    xfail_not_implemented("change_detection.detect_change_points_from_embeddings")