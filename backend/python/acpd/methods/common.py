"""Shared utilities for segmentation method implementations."""

from collections import Counter

import numpy as np

from ..data import normalize_segment_label


def build_fixed_segments(audio_length, num_segments):
    """Create equal-width segments over an audio interval."""
    segment_length = audio_length / num_segments
    return [
        {
            "start": idx * segment_length,
            "end": (idx + 1) * segment_length,
        }
        for idx in range(num_segments)
    ]


def build_segments_from_splits(audio_length, split_times, target_num_segments):
    """Construct a segment list from split candidates under a strict budget."""
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
    """Assign a dominant label to each segment from frame-level labels."""
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


def infer_segment_labels_from_probabilities(segments, centers, prototype_probabilities, prototype_labels):
    """Assign segment labels from soft prototype probabilities."""
    labels = []
    mean_positive_masses = []
    max_positive_masses = []
    centers_arr = np.array(centers, dtype=np.float64)

    if len(prototype_probabilities) == 0 or len(prototype_labels) == 0:
        return ["background"] * len(segments)

    group_indices = {}
    for proto_idx, label in enumerate(prototype_labels):
        normalized = normalize_segment_label(label)
        if normalized not in group_indices:
            group_indices[normalized] = []
        group_indices[normalized].append(proto_idx)

    positive_proto_indices = [idx for idx, label in enumerate(prototype_labels) if normalize_segment_label(label) != "background"]

    for segment_idx, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        segment_mid = (start + end) / 2.0

        if segment_idx == len(segments) - 1:
            mask = (centers_arr >= start) & (centers_arr <= end)
        else:
            mask = (centers_arr >= start) & (centers_arr < end)

        frame_indices = np.where(mask)[0]
        if len(frame_indices) == 0:
            nearest_idx = int(np.argmin(np.abs(centers_arr - segment_mid))) if len(centers_arr) > 0 else None
            if nearest_idx is None:
                labels.append("background")
                mean_positive_masses.append(0.0)
                max_positive_masses.append(0.0)
                continue

            nearest_probs = prototype_probabilities[nearest_idx]
            masses = {}
            for normalized_label, proto_indices in group_indices.items():
                masses[normalized_label] = float(np.sum(nearest_probs[proto_indices]))

            background_mass = masses.get("background", 0.0)
            positive_masses = {label: mass for label, mass in masses.items() if label != "background"}

            if positive_proto_indices:
                pos_mass = float(np.sum(nearest_probs[positive_proto_indices]))
            else:
                pos_mass = 0.0
            mean_positive_masses.append(pos_mass)
            max_positive_masses.append(pos_mass)

            if not positive_masses:
                labels.append("background")
            else:
                best_positive_label = max(positive_masses.items(), key=lambda item: item[1])[0]
                best_positive_mass = positive_masses[best_positive_label]
                total_positive_mass = float(sum(positive_masses.values()))
                if pos_mass >= 0.5:
                    labels.append(best_positive_label)
                elif (best_positive_mass >= background_mass or total_positive_mass > background_mass) and pos_mass >= 0.35:
                    labels.append(best_positive_label)
                else:
                    labels.append("background")
            continue

        masses = {}
        for normalized_label, proto_indices in group_indices.items():
            masses[normalized_label] = float(np.sum(prototype_probabilities[frame_indices][:, proto_indices]))

        background_mass = masses.get("background", 0.0)
        positive_masses = {label: mass for label, mass in masses.items() if label != "background"}

        if positive_proto_indices:
            per_frame_positive_mass = np.sum(prototype_probabilities[frame_indices][:, positive_proto_indices], axis=1)
            mean_positive = float(np.mean(per_frame_positive_mass))
            max_positive = float(np.max(per_frame_positive_mass))
        else:
            mean_positive = 0.0
            max_positive = 0.0

        mean_positive_masses.append(mean_positive)
        max_positive_masses.append(max_positive)

        if not positive_masses:
            labels.append("background")
            continue

        best_positive_label = max(positive_masses.items(), key=lambda item: item[1])[0]
        best_positive_mass = positive_masses[best_positive_label]
        total_positive_mass = float(sum(positive_masses.values()))

        if mean_positive >= 0.5 or max_positive >= 0.8:
            labels.append(best_positive_label)
        elif (best_positive_mass >= background_mass or total_positive_mass > background_mass) and mean_positive >= 0.35:
            labels.append(best_positive_label)
        else:
            labels.append("background")

    corrected = list(labels)
    for idx in range(1, len(corrected) - 1):
        prev_label = corrected[idx - 1]
        next_label = corrected[idx + 1]
        curr_label = corrected[idx]

        if curr_label != "background":
            continue
        if prev_label == "background" or next_label == "background":
            continue
        if prev_label != next_label:
            continue
        if mean_positive_masses[idx] >= 0.6 or max_positive_masses[idx] >= 0.85:
            corrected[idx] = prev_label

    return corrected
