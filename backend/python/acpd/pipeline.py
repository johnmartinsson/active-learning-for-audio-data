from collections import Counter

import numpy as np

from .data import (
    load_embeddings_and_timings,
    normalize_segment_label,
    to_centers,
)
from .methods import common as common_methods
from .registry import resolve_method_runner


PROBABILITY_WINDOW_SIZE = 1
EMBEDDING_WINDOW_SIZE = 1
CHANGE_POINT_PROMINENCE = 0.0


def build_fixed_segments(audio_length, num_segments):
    """Create equal-width segments over an audio interval.

    Parameters
    ----------
    audio_length : float
        Total audio duration in seconds.
    num_segments : int
        Number of segments to generate.

    Returns
    -------
    list[dict[str, float]]
        Segment dicts in the form ``{"start": t0, "end": t1}``.
    """
    return common_methods.build_fixed_segments(audio_length, num_segments)


def build_segments_from_splits(audio_length, split_times, target_num_segments):
    """Construct a segment list from split candidates under a strict budget.

    Parameters
    ----------
    audio_length : float
        Total audio duration in seconds.
    split_times : list[float]
        Candidate split positions in seconds.
    target_num_segments : int
        Requested number of output segments.

    Returns
    -------
    list[dict[str, float]]
        Exactly ``target_num_segments`` non-overlapping segments. Falls back to
        equal-width segmentation if constraints cannot be satisfied.
    """
    return common_methods.build_segments_from_splits(audio_length, split_times, target_num_segments)


def infer_segment_labels(segments, centers, frame_labels):
    """Assign a dominant label to each segment from frame-level labels.

    Parameters
    ----------
    segments : list[dict[str, float]]
        Segment boundaries.
    centers : list[float]
        Frame center times.
    frame_labels : list[str]
        Predicted label for each frame.

    Returns
    -------
    list[str]
        One label per segment using majority vote, with background suppressed
        when a non-background label exists.
    """
    return common_methods.infer_segment_labels(segments, centers, frame_labels)


def infer_segment_labels_from_probabilities(segments, centers, prototype_probabilities, prototype_labels):
    """Assign segment labels from soft prototype probabilities.

    Parameters
    ----------
    segments : list[dict[str, float]]
        Segment boundaries.
    centers : list[float]
        Frame center times.
    prototype_probabilities : np.ndarray
        Frame-by-prototype probability matrix with shape
        ``(n_frames, n_prototypes)``.
    prototype_labels : list[str]
        Label associated with each prototype column.

    Returns
    -------
    list[str]
        One suggested label per segment.

    Notes
    -----
    This method avoids a hard-assignment mismatch where the change-point curve
    can indicate strong foreground confidence while nearest-prototype labels
    flip to background for a subset of frames. It also includes a small
    consistency post-pass for short background islands.
    """
    return common_methods.infer_segment_labels_from_probabilities(
        segments,
        centers,
        prototype_probabilities,
        prototype_labels,
    )


def build_segments_response(payload):
    """Execute segmentation pipeline and return API response payload.

    Parameters
    ----------
    payload : dict
        Input payload from the Node API layer. Expected keys:
        - ``audio_length`` (float-like)
        - ``labeling_strategy_choice`` (str, e.g. ``"fixed"`` or ``"active"``)
        - ``requested_num_segments`` (int-like)
        - ``embeddings_path`` (str)
        - ``labels_dir`` (str)
        - ``embeddings_dir`` (str)
        - ``negative_clustering_method`` (str)
        - ``num_negative_clusters`` (int-like)

    Returns
    -------
    dict
        JSON-serializable segmentation response including segment boundaries,
        per-frame metadata, and backend diagnostics.
    """
    audio_length = float(payload["audio_length"])
    labeling_strategy_choice = payload.get("labeling_strategy_choice", "fixed")
    adapted_method = payload.get("adapted_method")
    requested_num_segments = int(payload.get("requested_num_segments", 10))
    embeddings_path = payload.get("embeddings_path", "")
    labels_dir = payload.get("labels_dir", "")
    embeddings_dir = payload.get("embeddings_dir", "")
    negative_clustering_method = payload.get("negative_clustering_method", "none")
    num_negative_clusters = int(payload.get("num_negative_clusters", 1))

    num_segments = max(1, requested_num_segments)
    if embeddings_path:
        try:
            timings_arr, query_embeddings = load_embeddings_and_timings(embeddings_path)
        except FileNotFoundError:
            timings_arr = np.empty((0, 2), dtype=np.float64)
            query_embeddings = np.empty((0, 0), dtype=np.float64)
    else:
        timings_arr = np.empty((0, 2), dtype=np.float64)
        query_embeddings = np.empty((0, 0), dtype=np.float64)

    timings_centers = to_centers(timings_arr) if len(timings_arr) > 0 else []

    context = {
        "audio_length": audio_length,
        "num_segments": num_segments,
        "query_embeddings": query_embeddings,
        "timings_arr": timings_arr,
        "timings_centers": timings_centers,
        "labels_dir": labels_dir,
        "embeddings_dir": embeddings_dir,
        "negative_clustering_method": negative_clustering_method,
        "num_negative_clusters": num_negative_clusters,
        "labeling_strategy_choice": labeling_strategy_choice,
    }

    method_runner, resolved_method, source_override = resolve_method_runner(adapted_method, labeling_strategy_choice)
    method_result = method_runner(
        context,
        {
            "probability_window_size": PROBABILITY_WINDOW_SIZE,
            "embedding_window_size": EMBEDDING_WINDOW_SIZE,
            "change_point_prominence": CHANGE_POINT_PROMINENCE,
        },
    )

    source = source_override or method_result["source"]
    segments = method_result["segments"]
    suggested_labels = method_result["suggested_labels"]
    probabilities = method_result["probabilities"]
    frame_labels = method_result["frame_labels"]
    change_scores = method_result["change_scores"]
    change_point_times = method_result["change_point_times"]
    prototype_summary = method_result["prototype_summary"]

    if prototype_summary and frame_labels:
        prototype_summary["frame_label_counts"] = dict(
            Counter(normalize_segment_label(label) for label in frame_labels)
        )

    response = {
        "segments": segments,
        "probabilities": probabilities,
        "timings": timings_centers,
        "suggestedLabels": suggested_labels,
        "frameLabels": frame_labels,
        "changeScores": change_scores,
        "changePointTimes": change_point_times,
        "backend": {
            "engine": "python",
            "strategy": labeling_strategy_choice,
            "adapted_method": adapted_method or resolved_method,
            "requested_num_segments": requested_num_segments,
            "returned_num_segments": len(segments),
            "module": "python.acpd.get_segments_cli",
            "source": source,
            "negative_clustering_method": negative_clustering_method,
            "num_negative_clusters": num_negative_clusters,
            "prototype_summary": prototype_summary,
            "acpd": {
                "probability_window_size": PROBABILITY_WINDOW_SIZE,
                "embedding_window_size": EMBEDDING_WINDOW_SIZE,
                "prominence": CHANGE_POINT_PROMINENCE,
                "num_change_points": max(0, len(segments) - 1),
            },
        },
    }

    return response