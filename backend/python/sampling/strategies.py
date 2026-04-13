import os
from typing import Callable

import numpy as np

from python.acpd.data import is_background_label, load_embeddings_and_timings, read_label_rows


def _binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    return -(probs * np.log2(probs) + (1.0 - probs) * np.log2(1.0 - probs))


def _presence_probabilities(
    embeddings: np.ndarray,
    presence_prototype: np.ndarray,
    absence_prototype: np.ndarray,
) -> np.ndarray:
    if len(embeddings) == 0:
        return np.array([], dtype=np.float64)

    presence_dist = np.linalg.norm(embeddings - presence_prototype[None, :], axis=1)
    absence_dist = np.linalg.norm(embeddings - absence_prototype[None, :], axis=1)

    logits = np.stack([-absence_dist, -presence_dist], axis=1)
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs[:, 1]


def _compute_legacy_binary_prototypes(
    labels_dir: str,
    embeddings_dir: str,
    embedding_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    random_prototype = lambda: rng.random(embedding_size, dtype=np.float64)

    if not labels_dir or not os.path.isdir(labels_dir):
        return random_prototype(), random_prototype()

    label_files = [name for name in os.listdir(labels_dir) if name.endswith(".txt")]
    if not label_files:
        return random_prototype(), random_prototype()

    presence_embeddings = []
    absence_embeddings = []

    for file_name in label_files:
        stem = os.path.splitext(file_name)[0]
        label_path = os.path.join(labels_dir, file_name)
        embeddings_path = os.path.join(embeddings_dir, f"{stem}.birdnet.embeddings.msgpack")

        if not os.path.exists(embeddings_path):
            continue

        rows = read_label_rows(label_path)
        if not rows:
            continue

        timings, embeddings = load_embeddings_and_timings(embeddings_path)
        if len(timings) == 0 or len(embeddings) == 0:
            continue

        centers = (timings[:, 0] + timings[:, 1]) / 2.0

        for start_time, end_time, label in rows:
            frame_mask = (centers >= float(start_time)) & (centers <= float(end_time))
            if not np.any(frame_mask):
                continue

            selected_embeddings = embeddings[frame_mask]
            if is_background_label(label):
                absence_embeddings.extend(selected_embeddings)
            else:
                presence_embeddings.extend(selected_embeddings)

    if presence_embeddings:
        presence_arr = np.asarray(presence_embeddings, dtype=np.float64)
        presence_prototype = np.mean(presence_arr, axis=0)
    else:
        presence_prototype = random_prototype()

    if absence_embeddings:
        absence_arr = np.asarray(absence_embeddings, dtype=np.float64)
        absence_prototype = np.mean(absence_arr, axis=0)
    else:
        absence_prototype = random_prototype()

    return presence_prototype, absence_prototype


def _score_files(
    file_names: list[str],
    embeddings_dir: str,
    scorer: Callable[[np.ndarray], float],
) -> list[tuple[str, float]]:
    scored_files = []
    for filename in file_names:
        embeddings_path = os.path.join(embeddings_dir, f"{filename}.birdnet.embeddings.msgpack")
        if not os.path.exists(embeddings_path):
            continue

        _, embeddings = load_embeddings_and_timings(embeddings_path)
        if len(embeddings) == 0:
            continue

        score = float(scorer(np.asarray(embeddings, dtype=np.float64)))
        scored_files.append((filename, score))

    return scored_files


def sample_files(payload: dict) -> list[str]:
    """Sample a batch of filenames according to the requested strategy."""
    strategy = str(payload.get("strategy", "random") or "random").strip().lower()
    batch_size = max(1, int(payload.get("batch_size", 1)))
    unlabeled_files = list(payload.get("unlabeled_files", []) or [])

    if not unlabeled_files:
        return []

    if strategy == "random":
        rng = np.random.default_rng()
        shuffled = list(unlabeled_files)
        rng.shuffle(shuffled)
        return shuffled[:batch_size]

    labels_dir = str(payload.get("labels_dir", "") or "")
    embeddings_dir = str(payload.get("embeddings_dir", "") or "")
    presence_prototype, absence_prototype = _compute_legacy_binary_prototypes(labels_dir, embeddings_dir)

    def presence_trace(emb: np.ndarray) -> np.ndarray:
        return _presence_probabilities(emb, presence_prototype, absence_prototype)

    if strategy == "uncertainty":
        scored = _score_files(
            unlabeled_files,
            embeddings_dir,
            lambda emb: float(np.mean(_binary_entropy(presence_trace(emb)))),
        )
        scored.sort(key=lambda item: item[1], reverse=True)
    elif strategy == "certainty":
        scored = _score_files(
            unlabeled_files,
            embeddings_dir,
            lambda emb: float(np.mean(_binary_entropy(presence_trace(emb)))),
        )
        scored.sort(key=lambda item: item[1])
    elif strategy == "high_probability":
        scored = _score_files(
            unlabeled_files,
            embeddings_dir,
            lambda emb: float(np.mean(presence_trace(emb))),
        )
        scored.sort(key=lambda item: item[1], reverse=True)
    else:
        rng = np.random.default_rng()
        shuffled = list(unlabeled_files)
        rng.shuffle(shuffled)
        return shuffled[:batch_size]

    return [filename for filename, _ in scored[:batch_size]]
