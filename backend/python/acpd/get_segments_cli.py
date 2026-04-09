import json
import os
import sys
from collections import Counter, defaultdict

import msgpack
import numpy as np


def build_fixed_segments(audio_length, num_segments):
    segment_length = audio_length / num_segments
    return [
        {
            "start": idx * segment_length,
            "end": (idx + 1) * segment_length,
        }
        for idx in range(num_segments)
    ]


def load_timings(embeddings_path):
    with open(embeddings_path, "rb") as f:
        payload = msgpack.unpackb(f.read(), raw=False)

    timings = payload.get("timings", [])
    return [[float(t[0]), float(t[1])] for t in timings if len(t) == 2]


def load_embeddings_and_timings(embeddings_path):
    with open(embeddings_path, "rb") as f:
        payload = msgpack.unpackb(f.read(), raw=False)

    timings = payload.get("timings", [])
    embeddings = payload.get("embeddings", [])

    timings = np.array([[float(t[0]), float(t[1])] for t in timings if len(t) == 2], dtype=np.float64)
    embeddings = np.array(embeddings, dtype=np.float64)

    if len(timings) != len(embeddings):
        n = min(len(timings), len(embeddings))
        timings = timings[:n]
        embeddings = embeddings[:n]

    return timings, embeddings


def to_centers(timings):
    if isinstance(timings, np.ndarray):
        return ((timings[:, 0] + timings[:, 1]) / 2.0).astype(np.float64).tolist()
    return [float((start + end) / 2.0) for start, end in timings]


def normalize_label(label):
    return str(label or "").strip().lower()


def is_background_label(label):
    normalized = normalize_label(label)
    return normalized in ("", "background", "absence")


def read_label_rows(label_path):
    rows = []
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        return rows

    has_header = lines[0].lower().startswith("start_time")
    start_idx = 1 if has_header else 0

    for line in lines[start_idx:]:
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            start_time = float(parts[0])
            end_time = float(parts[1])
            label = ",".join(parts[2:]).strip()
            rows.append((start_time, end_time, label))
        except ValueError:
            continue

    return rows


def collect_labeled_embeddings(labels_dir, embeddings_dir):
    class_embeddings = defaultdict(list)
    background_embeddings = []

    if not labels_dir or not os.path.isdir(labels_dir):
        return class_embeddings, background_embeddings

    for file_name in os.listdir(labels_dir):
        if not file_name.endswith(".txt"):
            continue

        stem = os.path.splitext(file_name)[0]
        label_path = os.path.join(labels_dir, file_name)
        file_embeddings_path = os.path.join(embeddings_dir, f"{stem}.birdnet.embeddings.msgpack")

        if not os.path.exists(file_embeddings_path):
            continue

        rows = read_label_rows(label_path)
        if not rows:
            continue

        timings, embeddings = load_embeddings_and_timings(file_embeddings_path)
        if len(timings) == 0:
            continue
        centers = (timings[:, 0] + timings[:, 1]) / 2.0

        for start_time, end_time, label in rows:
            frame_mask = (centers >= start_time) & (centers <= end_time)
            if not np.any(frame_mask):
                continue
            selected = embeddings[frame_mask]

            normalized = normalize_label(label)
            if is_background_label(normalized):
                background_embeddings.extend(selected)
            else:
                class_embeddings[normalized].extend(selected)

    return class_embeddings, background_embeddings


def compute_mean_prototypes(class_embeddings):
    prototypes = {}
    for label, vectors in class_embeddings.items():
        if not vectors:
            continue
        arr = np.array(vectors, dtype=np.float64)
        prototypes[label] = arr.mean(axis=0)
    return prototypes


def run_kmeans(data, k, max_iter=25):
    if len(data) == 0:
        return np.empty((0, 0), dtype=np.float64)

    k = max(1, min(int(k), len(data)))
    rng = np.random.default_rng(seed=0)
    init_idx = rng.choice(len(data), size=k, replace=False)
    centroids = data[init_idx].copy()

    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)

        updated = centroids.copy()
        for cluster_idx in range(k):
            members = data[assignments == cluster_idx]
            if len(members) > 0:
                updated[cluster_idx] = members.mean(axis=0)

        if np.allclose(updated, centroids):
            break
        centroids = updated

    return centroids


def compute_background_prototypes(background_vectors, method, num_clusters):
    if len(background_vectors) == 0:
        return {}

    arr = np.array(background_vectors, dtype=np.float64)
    normalized_method = str(method or "none").strip().lower()

    if normalized_method == "kmeans" and int(num_clusters) > 1:
        centroids = run_kmeans(arr, int(num_clusters))
        return {f"background_{i}": centroid for i, centroid in enumerate(centroids)}

    return {"background": arr.mean(axis=0)}


def softmax(values):
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    sums = np.sum(exp_vals, axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return exp_vals / sums


def build_switch_segments(audio_length, centers, frame_labels):
    if len(centers) == 0:
        return [{"start": 0.0, "end": float(audio_length)}]

    boundaries = [0.0]
    for idx in range(1, len(frame_labels)):
        if frame_labels[idx] != frame_labels[idx - 1]:
            split = float((centers[idx - 1] + centers[idx]) / 2.0)
            split = min(max(split, 0.0), float(audio_length))
            if split > boundaries[-1]:
                boundaries.append(split)

    boundaries.append(float(audio_length))

    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            segments.append({"start": start, "end": end})
    return segments


def build_switch_segments_with_budget(audio_length, centers, frame_labels, target_num_segments, embeddings=None):
    if target_num_segments <= 1:
        return [{"start": 0.0, "end": float(audio_length)}]

    if len(centers) == 0:
        return build_fixed_segments(audio_length, target_num_segments)

    target_splits = target_num_segments - 1
    candidates = []

    for idx in range(1, len(frame_labels)):
        if frame_labels[idx] == frame_labels[idx - 1]:
            continue

        split = float((centers[idx - 1] + centers[idx]) / 2.0)
        split = min(max(split, 0.0), float(audio_length))

        if embeddings is not None and idx < len(embeddings):
            score = float(np.linalg.norm(embeddings[idx] - embeddings[idx - 1]))
        else:
            score = 1.0

        candidates.append((split, score))

    unique_candidates = {}
    for split, score in candidates:
        if split not in unique_candidates or score > unique_candidates[split]:
            unique_candidates[split] = score

    sorted_candidates = sorted(unique_candidates.items(), key=lambda x: x[1], reverse=True)
    selected_splits = sorted([split for split, _ in sorted_candidates[:target_splits]])

    if len(selected_splits) < target_splits:
        needed = target_splits - len(selected_splits)
        fallback_splits = np.linspace(0.0, float(audio_length), needed + 2)[1:-1].tolist()
        for split in fallback_splits:
            clamped = min(max(float(split), 0.0), float(audio_length))
            if clamped not in selected_splits:
                selected_splits.append(clamped)
        selected_splits = sorted(selected_splits[:target_splits])

    boundaries = [0.0] + selected_splits + [float(audio_length)]

    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            segments.append({"start": float(start), "end": float(end)})

    if len(segments) != target_num_segments:
        return build_fixed_segments(audio_length, target_num_segments)

    return segments


def build_embedding_change_segments(audio_length, timings_arr, embeddings, target_num_segments):
    if target_num_segments <= 1 or len(embeddings) < 2:
        return [{"start": 0.0, "end": float(audio_length)}]

    centers = ((timings_arr[:, 0] + timings_arr[:, 1]) / 2.0).astype(np.float64)
    split_scores = np.linalg.norm(embeddings[1:] - embeddings[:-1], axis=1)
    n_splits = min(target_num_segments - 1, len(split_scores))

    top_indices = np.argsort(split_scores)[-n_splits:]
    split_times = sorted(float((centers[idx] + centers[idx + 1]) / 2.0) for idx in top_indices)

    boundaries = [0.0] + split_times + [float(audio_length)]
    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            segments.append({"start": float(start), "end": float(end)})

    if len(segments) < target_num_segments:
        # Fill up to requested budget with uniform splits if there are not enough frame changes.
        return build_fixed_segments(audio_length, target_num_segments)

    return segments


def normalize_segment_label(label):
    normalized = normalize_label(label)
    if normalized.startswith("background"):
        return "background"
    return normalized or "background"


def infer_segment_labels(segments, centers, frame_labels):
    labels = []
    centers_arr = np.array(centers, dtype=np.float64)

    for seg_idx, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])

        if seg_idx == len(segments) - 1:
            mask = (centers_arr >= start) & (centers_arr <= end)
        else:
            mask = (centers_arr >= start) & (centers_arr < end)

        chosen = [normalize_segment_label(frame_labels[i]) for i in np.where(mask)[0]]

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


def build_timing_aware_segments(audio_length, num_segments, timings):
    if num_segments <= 1:
        return [{"start": 0.0, "end": float(audio_length)}]

    centers = to_centers(timings)
    if len(centers) < 2:
        return build_fixed_segments(audio_length, num_segments)

    positions = np.linspace(0, len(centers) - 1, num_segments + 1)[1:-1]
    split_times = np.interp(positions, np.arange(len(centers)), centers)

    boundaries = [0.0]
    for split in split_times:
        clamped = min(max(float(split), 0.0), float(audio_length))
        if clamped > boundaries[-1]:
            boundaries.append(clamped)
    boundaries.append(float(audio_length))

    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            segments.append({"start": start, "end": end})

    if not segments:
        return build_fixed_segments(audio_length, num_segments)

    return segments


def main():
    payload = json.loads(sys.stdin.read())

    audio_length = float(payload["audio_length"])
    labeling_strategy_choice = payload.get("labeling_strategy_choice", "fixed")
    requested_num_segments = int(payload.get("requested_num_segments", 10))
    embeddings_path = payload.get("embeddings_path", "")
    labels_dir = payload.get("labels_dir", "")
    embeddings_dir = payload.get("embeddings_dir", "")
    negative_clustering_method = payload.get("negative_clustering_method", "none")
    num_negative_clusters = int(payload.get("num_negative_clusters", 1))

    num_segments = max(1, requested_num_segments)

    timings = []
    timings_centers = []
    source = "fixed-fallback"
    probabilities = []
    frame_labels = []
    prototype_summary = {}

    if embeddings_path and os.path.exists(embeddings_path):
        timings_arr, query_embeddings = load_embeddings_and_timings(embeddings_path)
        timings = timings_arr.tolist()
        timings_centers = to_centers(timings_arr)

        if labeling_strategy_choice == "fixed":
            segments = build_fixed_segments(audio_length, num_segments)
            suggested_labels = ["background"] * len(segments)
            source = "fixed-even"
        elif labeling_strategy_choice == "active" and len(query_embeddings) > 0:
            class_vectors, background_vectors = collect_labeled_embeddings(labels_dir, embeddings_dir)
            positive_prototypes = compute_mean_prototypes(class_vectors)
            background_prototypes = compute_background_prototypes(
                background_vectors,
                negative_clustering_method,
                num_negative_clusters,
            )

            # If there are no positive labels yet, use embedding-change segmentation as a simple
            # adaptive fallback and keep all segment suggestions as background.
            if len(positive_prototypes) == 0:
                segments = build_embedding_change_segments(
                    audio_length,
                    timings_arr,
                    query_embeddings,
                    num_segments,
                )
                suggested_labels = ["background"] * len(segments)
                source = "embedding-change-points"
                probabilities = []
                frame_labels = []
                prototype_summary = {
                    "positive_labels": [],
                    "num_positive_prototypes": 0,
                    "num_background_prototypes": len(background_prototypes),
                }
            else:
                # If no background labels exist yet, derive temporary background prototypes from
                # the current file so positive-vs-background distances are still computable.
                if len(background_prototypes) == 0:
                    fallback_clusters = max(1, num_negative_clusters)
                    fallback_method = "kmeans" if fallback_clusters > 1 else "none"
                    background_prototypes = compute_background_prototypes(
                        query_embeddings,
                        fallback_method,
                        fallback_clusters,
                    )

                all_prototypes = {**positive_prototypes, **background_prototypes}
                prototype_labels = list(all_prototypes.keys())

                if len(all_prototypes) >= 2:
                    prototype_matrix = np.stack([all_prototypes[label] for label in prototype_labels], axis=0)
                    distances = np.linalg.norm(query_embeddings[:, None, :] - prototype_matrix[None, :, :], axis=2)
                    nearest_idx = np.argmin(distances, axis=1)
                    frame_labels = [prototype_labels[idx] for idx in nearest_idx]

                    probs = softmax(-distances)
                    positive_indices = [i for i, label in enumerate(prototype_labels) if not is_background_label(label)]
                    if positive_indices:
                        probabilities = np.sum(probs[:, positive_indices], axis=1).tolist()
                    else:
                        probabilities = [0.0] * len(frame_labels)

                    segments = build_switch_segments_with_budget(
                        audio_length,
                        timings_centers,
                        frame_labels,
                        num_segments,
                        embeddings=query_embeddings,
                    )
                    source = "prototype-switches"
                    suggested_labels = infer_segment_labels(segments, timings_centers, frame_labels)
                else:
                    segments = build_timing_aware_segments(audio_length, num_segments, timings)
                    suggested_labels = ["background"] * len(segments)
                    source = "msgpack-timings-no-prototypes"

                prototype_summary = {
                    "positive_labels": sorted(list(positive_prototypes.keys())),
                    "num_positive_prototypes": len(positive_prototypes),
                    "num_background_prototypes": len(background_prototypes),
                }
        else:
            segments = build_timing_aware_segments(audio_length, num_segments, timings)
            suggested_labels = ["background"] * len(segments)
            source = "msgpack-timings"
    else:
        segments = build_fixed_segments(audio_length, num_segments)
        suggested_labels = ["background"] * len(segments)

    response = {
        "segments": segments,
        "probabilities": probabilities,
        "timings": timings_centers,
        "suggestedLabels": suggested_labels,
        "frameLabels": frame_labels,
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
        },
    }

    print(json.dumps(response))


if __name__ == "__main__":
    main()
