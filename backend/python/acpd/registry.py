"""Method registry for A-CPD segmentation strategies."""

from .methods import adaptive, fixed


METHODS = {
    "fixed": fixed.run,
    "active": adaptive.run,
    "acpd_probability": adaptive.run,
}


def resolve_method_runner(adapted_method, labeling_strategy_choice):
    """Resolve a method implementation from payload preferences.

    Parameters
    ----------
    adapted_method : str | None
        Optional explicit method key from request payload.
    labeling_strategy_choice : str
        Legacy strategy key (for backward compatibility).

    Returns
    -------
    tuple[callable, str, str | None]
        ``(runner, resolved_method_name, source_override)``.
    """
    if adapted_method:
        normalized = str(adapted_method).strip().lower()
        runner = METHODS.get(normalized)
        if runner is not None:
            return runner, normalized, None
        return METHODS["fixed"], "fixed", "fixed-unknown-method"

    normalized_strategy = str(labeling_strategy_choice or "fixed").strip().lower()
    if normalized_strategy == "active":
        return METHODS["active"], "active", None
    if normalized_strategy == "fixed":
        return METHODS["fixed"], "fixed", None
    return METHODS["fixed"], "fixed", "fixed-unknown-strategy"
