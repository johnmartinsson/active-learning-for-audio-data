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
    assert peak_times == [1.0, 2.0]


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


def test_detect_change_points_from_probabilities_matches_reported_example():
    """Reproduce the user-reported real example exactly.

    This locks in the observed output so we can reason about regressions with a
    fixed input sequence.
    """
    probabilities = [
        0.03346089794066871,
        0.9953370587841301,
        0.9981400190912659,
        0.03920332600470586,
        0.10303552823879579,
        0.9972243202377556,
        0.9927535082015735,
        0.35682940490420034,
        0.9969657155382845,
        0.030051804893206416,
        0.9980304512508876,
        0.8400474152392019,
        0.7716108990544406,
        0.20079333272200373,
    ]
    timings_centers = [
        0.5,
        2.6,
        4.7,
        6.800000000000001,
        8.9,
        11.0,
        13.1,
        15.2,
        17.3,
        19.400000000000002,
        21.500000000000004,
        23.600000000000005,
        25.700000000000006,
        27.800000000000008,
    ]
    timings = np.asarray([[t - 1.05, t + 1.05] for t in timings_centers], dtype=np.float64)

    peak_times, scores, peak_indices, _ = change_detection.detect_change_points_from_probabilities(
        probabilities,
        timings,
        n_peaks=len(probabilities) - 1,
        prominence=0.0,
        window_size=1,
    )

    expected_scores = np.zeros(len(probabilities), dtype=np.float64)
    expected_scores[1:] = np.abs(np.diff(np.asarray(probabilities, dtype=np.float64)))
    expected_peak_indices = list(range(1, len(probabilities)))
    expected_peak_times = [timings[idx, 0] for idx in expected_peak_indices]

    assert np.allclose(scores, expected_scores.tolist())
    assert peak_indices == expected_peak_indices
    assert np.allclose(peak_times, expected_peak_times)


def test_detect_change_points_from_probabilities_alternating_sequence_detects_each_transition():
    """Alternating 0/1 probabilities should produce one boundary score per flip.

    This is the regression test for very short events where every adjacent pair
    flips class identity.
    """
    probabilities = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    timings = np.asarray([[float(i), float(i + 1)] for i in range(len(probabilities))], dtype=np.float64)

    peak_times, scores, peak_indices, prominences = change_detection.detect_change_points_from_probabilities(
        probabilities,
        timings,
        n_peaks=len(probabilities) - 1,
        prominence=0.0,
        window_size=1,
    )

    assert scores == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert peak_indices == [1, 2, 3, 4, 5, 6]
    assert peak_times == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert prominences == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]