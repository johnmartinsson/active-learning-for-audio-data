"""Legacy change-point detection script for probability traces.

This script remains for compatibility with the older Node sampling path.
It reads a JSON payload from stdin and prints a JSON list of selected
change-point indices.
"""

import json
import sys

import numpy as np
from scipy.signal import find_peaks


def euclidean_distance_score(x1, x2):
    """Compute Euclidean distance between two vectors.

    Parameters
    ----------
    x1 : np.ndarray
        First vector.
    x2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        L2 distance.
    """
    return np.sqrt(np.sum(np.power(x1 - x2, 2)))


def distance_past_and_future_averages(embeddings, distance_fn, offset=5, M=5):
    """Compute a change curve from past/future local means.

    Parameters
    ----------
    embeddings : np.ndarray
        Input sequence with shape ``(n_frames, n_features)``.
    distance_fn : Callable[[np.ndarray, np.ndarray], float]
        Distance function between local means.
    offset : int, default=5
        Gap between index and both context windows.
    M : int, default=5
        Window size for past and future averaging.

    Returns
    -------
    np.ndarray
        Change score per frame index.
    """
    ds = np.zeros(len(embeddings))

    for idx in range(M + offset, len(embeddings) - M - offset):
        past_start = idx - M - offset
        past_end = idx - offset
        future_start = idx + 1 + offset
        future_end = idx + M + 1 + offset

        past_mean = np.mean(embeddings[past_start:past_end, :], axis=0)
        future_mean = np.mean(embeddings[future_start:future_end, :], axis=0)

        distance = distance_fn(past_mean, future_mean)
        ds[idx] = distance

    return ds


def detect_change_points(probabilities, num_segments, prominence_threshold=0):
    """Detect top change-point indices from a scalar probability trace.

    Parameters
    ----------
    probabilities : list[float] | np.ndarray
        Foreground probability per frame.
    num_segments : int
        Target number of output segments. Number of change points is
        ``num_segments - 1``.
    prominence_threshold : float, default=0
        Minimum peak prominence for selection.

    Returns
    -------
    list[int]
        Sorted frame indices selected as change points.
    """
    probabilities = np.array(probabilities)
    probabilities = probabilities.reshape((len(probabilities), 1))
    ds = distance_past_and_future_averages(probabilities, euclidean_distance_score, offset=0, M=1)

    n_change_points = num_segments - 1
    peaks = find_peaks(ds, prominence=prominence_threshold)

    peak_indices = peaks[0]
    peak_prominences = peaks[1]["prominences"]
    ranked = sorted(list(zip(peak_indices, peak_prominences)), key=lambda x: x[1], reverse=True)

    top_n_peak_indices_sorted = sorted([int(x[0]) for x in ranked[:n_change_points]])
    return top_n_peak_indices_sorted


if __name__ == "__main__":
    """CLI entrypoint: read JSON payload from stdin and print JSON output."""
    input_data = json.loads(sys.stdin.read())
    probabilities = input_data["probabilities"]
    num_segments = input_data["num_segments"]
    prominence_threshold = input_data.get("prominence_threshold", 0)

    change_points = detect_change_points(probabilities, num_segments, prominence_threshold)
    print(json.dumps(change_points))
