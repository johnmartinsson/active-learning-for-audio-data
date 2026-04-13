import os
from collections import defaultdict

import msgpack
import numpy as np


def load_embeddings_and_timings(embeddings_path):
    """Load frame timings and embeddings from a BirdNET msgpack file.

    Parameters
    ----------
    embeddings_path : str
        Absolute or relative path to a ``*.birdnet.embeddings.msgpack`` file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple ``(timings, embeddings)`` where:
        - ``timings`` has shape ``(n_frames, 2)`` with ``[start, end]`` per frame.
        - ``embeddings`` has shape ``(n_frames, embedding_dim)``.

        If frame counts differ in the input payload, both arrays are truncated
        to the shortest available length.
    """
    with open(embeddings_path, "rb") as file_obj:
        payload = msgpack.unpackb(file_obj.read(), raw=False)

    timings = payload.get("timings", [])
    embeddings = payload.get("embeddings", [])

    timings = np.array([[float(t[0]), float(t[1])] for t in timings if len(t) == 2], dtype=np.float64)
    embeddings = np.array(embeddings, dtype=np.float64)

    if len(timings) != len(embeddings):
        n_frames = min(len(timings), len(embeddings))
        timings = timings[:n_frames]
        embeddings = embeddings[:n_frames]

    return timings, embeddings


def to_centers(timings):
    """Convert frame intervals to center times.

    Parameters
    ----------
    timings : np.ndarray | list[tuple[float, float]]
        Frame start and end times.

    Returns
    -------
    list[float]
        Center time for each frame.
    """
    if isinstance(timings, np.ndarray):
        return ((timings[:, 0] + timings[:, 1]) / 2.0).astype(np.float64).tolist()
    return [float((start + end) / 2.0) for start, end in timings]


def normalize_label(label):
    """Normalize a free-text label for internal processing.

    Parameters
    ----------
    label : Any
        Label-like value from user input or persisted annotation files.

    Returns
    -------
    str
        Lower-cased, stripped string representation.
    """
    return str(label or "").strip().lower()


def is_background_label(label):
    """Check whether a label should be treated as background.

    Parameters
    ----------
    label : Any
        Label value to test.

    Returns
    -------
    bool
        ``True`` if the label belongs to the background-like set.
    """
    normalized = normalize_label(label)
    return (
        normalized in ("", "background", "absence")
        or normalized.startswith("background")
        or normalized.startswith("absence")
    )


def normalize_segment_label(label):
    """Normalize predicted frame labels before segment-level voting.

    Parameters
    ----------
    label : Any
        Frame-level prototype label.

    Returns
    -------
    str
        Background labels collapse to ``"background"``; other labels are
        normalized and returned unchanged.
    """
    normalized = normalize_label(label)
    if normalized.startswith("background"):
        return "background"
    return normalized or "background"


def read_label_rows(label_path):
    """Read annotation rows from a label file.

    Parameters
    ----------
    label_path : str
        Path to a CSV-like text file with either:
        ``start_time,end_time,label`` header, or raw rows with this layout.

    Returns
    -------
    list[tuple[float, float, str]]
        Parsed rows as ``(start_time, end_time, label)``.
        Malformed lines are skipped.
    """
    rows = []
    with open(label_path, "r", encoding="utf-8") as file_obj:
        lines = [line.strip() for line in file_obj.readlines() if line.strip()]

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
    """Collect labeled embeddings from all project label files.

    Parameters
    ----------
    labels_dir : str
        Directory containing per-file annotation text files.
    embeddings_dir : str
        Directory containing BirdNET msgpack embeddings.

    Returns
    -------
    tuple[defaultdict[str, list[np.ndarray]], list[np.ndarray]]
        ``(class_embeddings, background_embeddings)`` where positive labels are
        grouped by class name and background-like frames are returned as a
        flat list.
    """
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

            selected_embeddings = embeddings[frame_mask]
            normalized = normalize_label(label)

            if is_background_label(normalized):
                background_embeddings.extend(selected_embeddings)
            else:
                class_embeddings[normalized].extend(selected_embeddings)

    return class_embeddings, background_embeddings