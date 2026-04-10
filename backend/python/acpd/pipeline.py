from collections import Counter

import numpy as np

from .change_detection import detect_change_points_from_embeddings, detect_change_points_from_probabilities
from .data import (
    collect_labeled_embeddings,
    is_background_label,
    load_embeddings_and_timings,
    normalize_segment_label,
    to_centers,
)
from .prototypes import (
    compute_background_prototypes,
    compute_mean_prototypes,
    infer_frame_labels_and_probabilities,
)


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
    segment_length = audio_length / num_segments
    return [
        {
            "start": idx * segment_length,
            "end": (idx + 1) * segment_length,
        }
        for idx in range(num_segments)
    ]


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
    if target_num_segments <= 1:
        return [{"start": 0.0, "end": float(audio_length)}]

    target_num_splits = target_num_segments - 1
    selected_splits = []
    seen = set()

    for split in split_times:
        clamped = min(max(float(split), 0.0), float(audio_length))
        if clamped <= 0.0 or clamped >= float(audio_length):
            continue
        if clamped in seen:
            continue
        selected_splits.append(clamped)
        seen.add(clamped)
        if len(selected_splits) == target_num_splits:
            break

    if len(selected_splits) < target_num_splits:
        fallback_splits = np.linspace(0.0, float(audio_length), target_num_splits + 2)[1:-1].tolist()
        for split in fallback_splits:
            clamped = min(max(float(split), 0.0), float(audio_length))
            if clamped in seen or clamped <= 0.0 or clamped >= float(audio_length):
                continue
            selected_splits.append(clamped)
            seen.add(clamped)
            if len(selected_splits) == target_num_splits:
                break

    boundaries = [0.0] + sorted(selected_splits[:target_num_splits]) + [float(audio_length)]
    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            segments.append({"start": float(start), "end": float(end)})

    if len(segments) != target_num_segments:
        return build_fixed_segments(audio_length, target_num_segments)

    return segments


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
    labels = []
    centers_arr = np.array(centers, dtype=np.float64)

    for segment_idx, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])

        if segment_idx == len(segments) - 1:
            mask = (centers_arr >= start) & (centers_arr <= end)
        else:
            mask = (centers_arr >= start) & (centers_arr < end)

        chosen = [normalize_segment_label(frame_labels[idx]) for idx in np.where(mask)[0]]
        if not chosen:
            labels.append("background")
            continue

        counts = Counter(chosen)
        if "background" in counts and len(counts) > 1:
            del counts["background"]

        if not counts:
            labels.append("background")
        else:
            labels.append(counts.most_common(1)[0][0])

    return labels


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
    requested_num_segments = int(payload.get("requested_num_segments", 10))
    embeddings_path = payload.get("embeddings_path", "")
    labels_dir = payload.get("labels_dir", "")
    embeddings_dir = payload.get("embeddings_dir", "")
    negative_clustering_method = payload.get("negative_clustering_method", "none")
    num_negative_clusters = int(payload.get("num_negative_clusters", 1))

    num_segments = max(1, requested_num_segments)
    timings_centers = []
    probabilities = []
    frame_labels = []
    prototype_summary = {}
    change_scores = []
    change_point_times = []
    source = "fixed-fallback"

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

    if len(query_embeddings) == 0 or labeling_strategy_choice == "fixed":
        segments = build_fixed_segments(audio_length, num_segments)
        suggested_labels = ["background"] * len(segments)
        source = "fixed-even" if labeling_strategy_choice == "fixed" else "fixed-fallback"
    elif labeling_strategy_choice == "active":
        class_vectors, background_vectors = collect_labeled_embeddings(labels_dir, embeddings_dir)
        positive_prototypes = compute_mean_prototypes(class_vectors)
        background_prototypes = compute_background_prototypes(
            background_vectors,
            negative_clustering_method,
            num_negative_clusters,
        )

        if not positive_prototypes:
            change_point_times, change_scores, _, prominences = detect_change_points_from_embeddings(
                query_embeddings,
                timings_arr,
                n_peaks=num_segments - 1,
                prominence=CHANGE_POINT_PROMINENCE,
                window_size=EMBEDDING_WINDOW_SIZE,
            )
            segments = build_segments_from_splits(audio_length, change_point_times, num_segments)
            suggested_labels = ["background"] * len(segments)
            source = "fcpd-embedding-curve"
            prototype_summary = {
                "positive_labels": [],
                "num_positive_prototypes": 0,
                "num_background_prototypes": len(background_prototypes),
                "peak_prominences": prominences,
            }
        else:
            if not background_prototypes:
                fallback_clusters = max(1, num_negative_clusters)
                fallback_method = "kmeans" if fallback_clusters > 1 else "none"
                background_prototypes = compute_background_prototypes(
                    query_embeddings,
                    fallback_method,
                    fallback_clusters,
                )

            frame_labels, probabilities, _, _ = infer_frame_labels_and_probabilities(
                query_embeddings,
                positive_prototypes,
                background_prototypes,
            )

            change_point_times, change_scores, _, prominences = detect_change_points_from_probabilities(
                probabilities,
                timings_arr,
                n_peaks=num_segments - 1,
                prominence=CHANGE_POINT_PROMINENCE,
                window_size=PROBABILITY_WINDOW_SIZE,
            )
            segments = build_segments_from_splits(audio_length, change_point_times, num_segments)
            suggested_labels = infer_segment_labels(segments, timings_centers, frame_labels)
            source = "acpd-probability-curve"
            prototype_summary = {
                "positive_labels": sorted(list(positive_prototypes.keys())),
                "num_positive_prototypes": len(positive_prototypes),
                "num_background_prototypes": len(background_prototypes),
                "peak_prominences": prominences,
            }
    else:
        segments = build_fixed_segments(audio_length, num_segments)
        suggested_labels = ["background"] * len(segments)
        source = "fixed-unknown-strategy"

    if prototype_summary and frame_labels:
        prototype_summary["frame_label_counts"] = Counter(normalize_segment_label(label) for label in frame_labels)

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