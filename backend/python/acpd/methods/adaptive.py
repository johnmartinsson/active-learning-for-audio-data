"""Active (adaptive) segmentation method implementations."""

from ..change_detection import detect_change_points_from_embeddings, detect_change_points_from_probabilities
from ..data import collect_labeled_embeddings
from ..prototypes import (
    compute_background_prototypes,
    compute_mean_prototypes,
    infer_frame_labels_and_probabilities,
)
from .common import build_fixed_segments, build_segments_from_splits, infer_segment_labels_from_probabilities


def run(context, config):
    """Run adaptive segmentation using prototype probabilities and CPD.

    Parameters
    ----------
    context : dict
        Precomputed pipeline context.
    config : dict
        Method configuration. Expected keys:
        ``change_point_prominence``, ``embedding_window_size``,
        ``probability_window_size``.

    Returns
    -------
    dict
        Method result payload.
    """
    audio_length = context["audio_length"]
    num_segments = context["num_segments"]
    query_embeddings = context["query_embeddings"]
    timings_arr = context["timings_arr"]
    timings_centers = context["timings_centers"]
    labels_dir = context["labels_dir"]
    embeddings_dir = context["embeddings_dir"]
    negative_clustering_method = context["negative_clustering_method"]
    num_negative_clusters = context["num_negative_clusters"]

    if len(query_embeddings) == 0:
        segments = build_fixed_segments(audio_length, num_segments)
        return {
            "segments": segments,
            "suggested_labels": ["background"] * len(segments),
            "source": "fixed-fallback",
            "probabilities": [],
            "frame_labels": [],
            "change_scores": [],
            "change_point_times": [],
            "prototype_summary": {},
        }

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
            prominence=config["change_point_prominence"],
            window_size=config["embedding_window_size"],
        )
        segments = build_segments_from_splits(audio_length, change_point_times, num_segments)
        return {
            "segments": segments,
            "suggested_labels": ["background"] * len(segments),
            "source": "fcpd-embedding-curve",
            "probabilities": [],
            "frame_labels": [],
            "change_scores": change_scores,
            "change_point_times": change_point_times,
            "prototype_summary": {
                "positive_labels": [],
                "num_positive_prototypes": 0,
                "num_background_prototypes": len(background_prototypes),
                "peak_prominences": prominences,
            },
        }

    if not background_prototypes:
        fallback_clusters = max(1, num_negative_clusters)
        fallback_method = "kmeans" if fallback_clusters > 1 else "none"
        background_prototypes = compute_background_prototypes(
            query_embeddings,
            fallback_method,
            fallback_clusters,
        )

    frame_labels, probabilities, prototype_probabilities, prototype_labels = infer_frame_labels_and_probabilities(
        query_embeddings,
        positive_prototypes,
        background_prototypes,
    )

    change_point_times, change_scores, _, prominences = detect_change_points_from_probabilities(
        probabilities,
        timings_arr,
        n_peaks=num_segments - 1,
        prominence=config["change_point_prominence"],
        window_size=config["probability_window_size"],
    )
    segments = build_segments_from_splits(audio_length, change_point_times, num_segments)
    suggested_labels = infer_segment_labels_from_probabilities(
        segments,
        timings_centers,
        prototype_probabilities,
        prototype_labels,
    )

    return {
        "segments": segments,
        "suggested_labels": suggested_labels,
        "source": "acpd-probability-curve",
        "probabilities": probabilities,
        "frame_labels": frame_labels,
        "change_scores": change_scores,
        "change_point_times": change_point_times,
        "prototype_summary": {
            "positive_labels": sorted(list(positive_prototypes.keys())),
            "num_positive_prototypes": len(positive_prototypes),
            "num_background_prototypes": len(background_prototypes),
            "peak_prominences": prominences,
        },
    }
