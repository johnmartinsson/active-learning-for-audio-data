"""Fixed-width segmentation method."""

from .common import build_fixed_segments


def run(context, config):
    """Run fixed-width segmentation.

    Parameters
    ----------
    context : dict
        Precomputed pipeline context.
    config : dict
        Method configuration (unused for this method).

    Returns
    -------
    dict
        Method result payload.
    """
    audio_length = context["audio_length"]
    num_segments = context["num_segments"]
    segments = build_fixed_segments(audio_length, num_segments)
    return {
        "segments": segments,
        "suggested_labels": ["background"] * len(segments),
        "source": "fixed-even",
        "probabilities": [],
        "frame_labels": [],
        "change_scores": [],
        "change_point_times": [],
        "prototype_summary": {},
    }
