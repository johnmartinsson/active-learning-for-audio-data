import numpy as np
from scipy.signal import find_peaks


def cosine_distance_score(x1, x2):
    """Compute cosine distance between two vectors.

    Parameters
    ----------
    x1 : np.ndarray
        First vector.
    x2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        Cosine distance in ``[0, 2]`` for finite inputs. Returns ``0.0`` when
        one of the vectors has zero norm.
    """
    denominator = np.linalg.norm(x1) * np.linalg.norm(x2)
    if denominator == 0:
        return 0.0
    return 1.0 - float(np.dot(x1, x2) / denominator)


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
    return float(np.sqrt(np.sum(np.square(x1 - x2))))


def distance_past_and_future_averages(sequence, distance_fn, offset=0, window_size=1):
    """Score each time index by comparing past and future local means.

    This is the core A-CPD/F-CPD curve primitive used for segmentation.

    Parameters
    ----------
    sequence : np.ndarray | list
        Input sequence with shape ``(n_frames,)`` or ``(n_frames, n_features)``.
    distance_fn : Callable[[np.ndarray, np.ndarray], float]
        Distance metric between past and future means.
    offset : int, default=0
        Gap between the current index and both windows.
    window_size : int, default=1
        Number of frames per past/future averaging window.

    Returns
    -------
    np.ndarray
        Change curve with one score per frame.
    """
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape((-1, 1))

    scores = np.zeros(len(values), dtype=np.float64)
    if len(values) == 0:
        return scores

    left_margin = window_size + offset
    right_margin = window_size + offset

    for idx in range(left_margin, len(values) - right_margin):
        past_start = idx - window_size - offset
        past_end = idx - offset
        future_start = idx + 1 + offset
        future_end = idx + window_size + 1 + offset

        past_mean = np.mean(values[past_start:past_end, :], axis=0)
        future_mean = np.mean(values[future_start:future_end, :], axis=0)
        scores[idx] = distance_fn(past_mean, future_mean)

    return scores


def distance_adjacent_windows_at_boundaries(sequence, distance_fn, window_size=1):
    """Score frame boundaries by comparing adjacent window means.

    For each boundary between frames ``i`` and ``i+1``, this computes a score
    from the mean of the ``window_size`` frames ending at ``i`` and the mean
    of the ``window_size`` frames starting at ``i+1``.

    The returned score array has one value per frame index, with boundary
    scores stored at the RIGHT frame index (``i+1``). Index ``0`` is always
    ``0.0`` because there is no left boundary for the first frame.

    Parameters
    ----------
    sequence : np.ndarray | list
        Input sequence with shape ``(n_frames,)`` or ``(n_frames, n_features)``.
    distance_fn : Callable[[np.ndarray, np.ndarray], float]
        Distance metric between adjacent window means.
    window_size : int, default=1
        Number of frames per adjacent averaging window.

    Returns
    -------
    np.ndarray
        Boundary change curve with one score per frame index.
    """
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape((-1, 1))

    n_frames = len(values)
    scores = np.zeros(n_frames, dtype=np.float64)
    if n_frames == 0:
        return scores

    if window_size <= 0:
        raise ValueError("window_size must be >= 1")

    # Boundary i|i+1 exists for i in [window_size-1, n_frames-window_size-1].
    for i in range(window_size - 1, n_frames - window_size):
        left_start = i - window_size + 1
        left_end = i + 1
        right_start = i + 1
        right_end = i + 1 + window_size

        left_mean = np.mean(values[left_start:left_end, :], axis=0)
        right_mean = np.mean(values[right_start:right_end, :], axis=0)
        scores[i + 1] = distance_fn(left_mean, right_mean)

    return scores


def rank_change_point_peaks(scores, prominence, n_peaks):
    """Select the most prominent change-point peaks.

    Parameters
    ----------
    scores : np.ndarray | list[float]
        Change curve values.
    prominence : float
        Minimum prominence passed to ``scipy.signal.find_peaks``.
    n_peaks : int
        Maximum number of peaks to return.

    Returns
    -------
    tuple[list[int], list[float]]
        Sorted peak indices and corresponding selected peak prominences.
    """
    if len(scores) == 0 or n_peaks <= 0:
        return [], []

    peaks = find_peaks(scores, prominence=prominence)
    peak_indices = peaks[0]
    peak_prominences = peaks[1].get("prominences", np.array([], dtype=np.float64))

    if len(peak_indices) == 0:
        return [], []

    ranked = sorted(zip(peak_indices, peak_prominences), key=lambda item: item[1], reverse=True)
    selected = ranked[:n_peaks]
    selected_indices = sorted(int(idx) for idx, _ in selected)
    selected_prominences = [float(prom) for idx, prom in selected if int(idx) in selected_indices]
    return selected_indices, selected_prominences


def rank_change_point_boundaries(scores, min_score, n_peaks):
    """Select strongest frame-boundary change scores.

    Unlike local-peak ranking, this keeps the top-scoring boundaries directly.
    This avoids missing rapid alternations where adjacent boundaries can all be
    informative but do not form isolated local maxima.

    Parameters
    ----------
    scores : np.ndarray | list[float]
        Boundary score values aligned to frame indices.
    min_score : float
        Minimum boundary score to keep.
    n_peaks : int
        Maximum number of boundaries to return.

    Returns
    -------
    tuple[list[int], list[float]]
        Sorted selected boundary indices and corresponding scores.
    """
    values = np.asarray(scores, dtype=np.float64)
    if len(values) == 0 or n_peaks <= 0:
        return [], []

    # Index 0 cannot represent a valid left-right boundary.
    candidates = [
        (idx, float(values[idx]))
        for idx in range(1, len(values))
        if float(values[idx]) > float(min_score)
    ]
    if not candidates:
        return [], []

    ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))
    selected = ranked[:n_peaks]
    selected_indices = sorted(idx for idx, _ in selected)
    score_by_idx = {idx: score for idx, score in selected}
    selected_scores = [score_by_idx[idx] for idx in selected_indices]
    return selected_indices, selected_scores


def peak_times_from_indices(timings, peak_indices):
    """Map peak frame indices to real-valued boundary times.

    Parameters
    ----------
    timings : np.ndarray
        Frame timings with shape ``(n_frames, 2)``.
    peak_indices : list[int]
        Indices selected on the change curve.

    Returns
    -------
    list[float]
        Start times of selected frames, which correspond to the boundary before
        each selected index when scores are boundary-aligned.
    """
    if len(peak_indices) == 0:
        return []

    peak_timings = timings[peak_indices, 0]
    return [float(time_value) for time_value in peak_timings.tolist()]


def detect_change_points_from_probabilities(probabilities, timings, n_peaks, prominence=0.0, window_size=1):
    """Detect change points from a scalar foreground probability trace.

    Parameters
    ----------
    probabilities : list[float] | np.ndarray
        Foreground probability per frame.
    timings : np.ndarray
        Frame timings with shape ``(n_frames, 2)``.
    n_peaks : int
        Number of change points to keep.
    prominence : float, default=0.0
        Peak prominence threshold.
    window_size : int, default=1
        Local averaging window size for the change curve.

    Returns
    -------
    tuple[list[float], list[float], list[int], list[float]]
        ``(peak_times, curve_scores, peak_indices, peak_prominences)``.
    """
    scores = distance_adjacent_windows_at_boundaries(
        np.asarray(probabilities, dtype=np.float64),
        distance_fn=euclidean_distance_score,
        window_size=window_size,
    )
    peak_indices, prominences = rank_change_point_boundaries(scores, min_score=prominence, n_peaks=n_peaks)
    return peak_times_from_indices(timings, peak_indices), scores.tolist(), peak_indices, prominences


def detect_change_points_from_embeddings(embeddings, timings, n_peaks, prominence=0.0, window_size=1):
    """Detect change points directly from embedding trajectories.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding sequence with shape ``(n_frames, embedding_dim)``.
    timings : np.ndarray
        Frame timings with shape ``(n_frames, 2)``.
    n_peaks : int
        Number of change points to keep.
    prominence : float, default=0.0
        Minimum boundary score threshold.
    window_size : int, default=1
        Local averaging window size for the change curve.

    Returns
    -------
    tuple[list[float], list[float], list[int], list[float]]
        ``(peak_times, curve_scores, peak_indices, boundary_scores)``.
    """
    scores = distance_adjacent_windows_at_boundaries(
        embeddings,
        distance_fn=cosine_distance_score,
        window_size=window_size,
    )
    peak_indices, prominences = rank_change_point_boundaries(scores, min_score=prominence, n_peaks=n_peaks)
    return peak_times_from_indices(timings, peak_indices), scores.tolist(), peak_indices, prominences