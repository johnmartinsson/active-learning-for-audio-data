# A-CPD Segmentation Notes

This document explains how the current Python segmentation pipeline works in:

- `backend/python/acpd/pipeline.py`
- `backend/python/acpd/registry.py`
- `backend/python/acpd/methods/`
- `backend/python/acpd/prototypes.py`
- `backend/python/acpd/change_detection.py`
- `backend/python/acpd/data.py`

It focuses on how multiclass prototypes are used to produce one foreground probability curve for A-CPD style change-point detection.

## Summary

The pipeline uses multiclass prototypes internally, then collapses them to a single binary foreground-vs-background probability trace for segmentation.

In short:

1. Build positive class prototypes and one or more background prototypes.
2. Compute per-frame distances to all prototypes.
3. Convert distances to prototype probabilities with softmax.
4. Sum probability mass across all non-background prototypes to get one foreground curve.
5. Run A-CPD style change-point detection on that curve.
6. Build segments from top change points under the requested segment budget.
7. Suggest a label per segment using soft probability mass aggregation.

## Method Registry and Orchestration

The pipeline now uses a simple method registry:

- `registry.py` resolves a method runner from either:
	- `adapted_method` (explicit selector), or
	- legacy `labeling_strategy_choice` (backward compatibility).
- Method implementations live in `methods/`.

Current methods:

- `methods/fixed.py`: equal-width segmentation
- `methods/adaptive.py`: prototype-based adaptive segmentation (A-CPD/F-CPD behavior)
- `methods/common.py`: shared segment utilities and soft label suggestion helpers

`pipeline.py` remains the orchestration layer:

1. Parse and normalize payload.
2. Load embeddings/timings once.
3. Build shared context.
4. Resolve and execute method.
5. Assemble stable API response shape.

## Detailed Formulation

Let:

- `x_t` be the embedding at frame `t`
- `p_1 ... p_K` be positive class prototypes
- `b_1 ... b_M` be background prototypes
- `c_j` be all prototypes concatenated (`K + M` total)

### 1) Distance to all prototypes

For each frame and prototype:

`d_{t,j} = || x_t - c_j ||_2`

Implemented in `infer_frame_labels_and_probabilities` in `prototypes.py`.

### 2) Softmax over negative distances

Convert distances into prototype posteriors:

`pi_{t,j} = exp(-d_{t,j}) / sum_l exp(-d_{t,l})`

This gives a probability distribution across all prototypes at each frame.

### 3) Collapse to one foreground probability curve

Let `P` be the index set of non-background prototypes. The foreground mass is:

`p_fg(t) = sum_{j in P} pi_{t,j}`

This is the single curve sent to change-point detection.

## Change-Point Curve (A-CPD Style)

Given scalar foreground sequence `p_fg(t)`, compute a change score at each frame by comparing local past and future means.

With window size `M` and offset `o`:

- `past_mean(t) = mean(p_fg[t-M-o : t-o])`
- `future_mean(t) = mean(p_fg[t+1+o : t+1+o+M])`
- `g(t) = distance(past_mean(t), future_mean(t))`

For scalar curves, Euclidean distance is used.

Peaks in `g(t)` are ranked by prominence and the top `num_segments - 1` become segment boundaries.

Implemented in `change_detection.py`:

- `distance_past_and_future_averages`
- `detect_change_points_from_probabilities`
- `rank_change_point_peaks`

## Segment Construction

`build_segments_from_splits` in `pipeline.py` creates exactly the requested number of segments when possible:

- clamp split times into `[0, audio_length]`
- deduplicate
- select top splits by ranking order
- if insufficient splits, add uniform fallback splits
- if still inconsistent, fallback to fixed equal-width segmentation

## Segment Label Suggestion Logic

Suggested labels are now computed from soft prototype probabilities, not hard nearest-prototype labels.

Why this matters:

- Segments are found from the soft foreground curve.
- Hard labels can disagree locally with soft mass (especially around plateaus, ties, or clustered backgrounds).
- This mismatch previously caused cases where a middle segment was marked background even though the foreground curve stayed high across it.

Current approach (`infer_segment_labels_from_probabilities`):

1. For each segment, collect contained frame indices.
2. Sum probability mass per normalized label group.
	- Example: `background`, `background_0`, `background_1` all collapse to `background`.
3. Pick strongest positive label when positive mass dominates background (or total positive mass exceeds background).
4. Otherwise suggest `background`.

This aligns label suggestion with the same soft evidence used for boundary detection.

## No-Positive-Label Fallback

If no positive prototypes are available yet:

- Use embedding trajectory change detection (`detect_change_points_from_embeddings`) instead of probability-curve CPD.
- Keep suggestions as `background`.

This corresponds to the source tag `fcpd-embedding-curve` in API responses.

## Why This Is a Good Fit

This design keeps the spirit of the original A-CPD workflow while extending to multiclass annotation:

- The core segmentation mechanism still runs on a single scalar change curve.
- Multiclass information is preserved upstream through multiple prototypes.
- Collapsing to foreground mass gives robust boundaries without forcing class-transition boundaries.
- Clustered background prototypes better model heterogeneous negatives while staying compatible with a binary boundary signal.

## Tradeoff and Possible Extension

Tradeoff:

- If class A changes to class B but total foreground mass remains stable, this may not create a boundary.

Possible extension:

- Add hybrid boundary scoring that combines foreground-mass CPD with class-distribution CPD.
- Keep this as an opt-in mode to preserve current behavior.

## Response Fields to Inspect

Useful fields in the CLI/API JSON response:

- `probabilities`: foreground probability curve (`p_fg`)
- `changeScores`: change curve values
- `changePointTimes`: selected split times
- `segments`: final segment boundaries
- `suggestedLabels`: per-segment class suggestions
- `backend.source`: which segmentation path was used
- `backend.prototype_summary`: prototype counts and labels

