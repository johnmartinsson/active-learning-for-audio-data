import numpy as np

from .data import is_background_label


def compute_mean_prototypes(class_embeddings):
    """Compute one mean prototype vector per positive class.

    Parameters
    ----------
    class_embeddings : dict[str, list[np.ndarray]]
        Mapping from class name to list of embedding vectors.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from class name to mean embedding vector.
    """
    prototypes = {}
    for label, vectors in class_embeddings.items():
        if not vectors:
            continue
        vectors_arr = np.array(vectors, dtype=np.float64)
        prototypes[label] = vectors_arr.mean(axis=0)
    return prototypes


def run_kmeans(data, k, max_iter=25):
    """Run deterministic k-means clustering over background embeddings.

    Parameters
    ----------
    data : np.ndarray
        Array of shape ``(n_samples, embedding_dim)``.
    k : int
        Requested number of clusters.
    max_iter : int, default=25
        Maximum number of optimization iterations.

    Returns
    -------
    np.ndarray
        Cluster centroids of shape ``(k_eff, embedding_dim)`` where ``k_eff``
        is clamped to ``[1, n_samples]``.
    """
    if len(data) == 0:
        return np.empty((0, 0), dtype=np.float64)

    k = max(1, min(int(k), len(data)))
    rng = np.random.default_rng(seed=0)
    init_indices = rng.choice(len(data), size=k, replace=False)
    centroids = data[init_indices].copy()

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
    """Build background prototype set from labeled background embeddings.

    Parameters
    ----------
    background_vectors : list[np.ndarray]
        Background-labeled embedding vectors.
    method : str
        Clustering method, currently ``"none"`` or ``"kmeans"``.
    num_clusters : int
        Requested number of background clusters for k-means.

    Returns
    -------
    dict[str, np.ndarray]
        One or more background prototype vectors keyed by prototype name.
    """
    if len(background_vectors) == 0:
        return {}

    background_arr = np.array(background_vectors, dtype=np.float64)
    normalized_method = str(method or "none").strip().lower()

    if normalized_method == "kmeans" and int(num_clusters) > 1:
        centroids = run_kmeans(background_arr, int(num_clusters))
        return {f"background_{idx}": centroid for idx, centroid in enumerate(centroids)}

    return {"background": background_arr.mean(axis=0)}


def softmax(values):
    """Compute row-wise softmax probabilities.

    Parameters
    ----------
    values : np.ndarray
        Input scores with shape ``(n_frames, n_prototypes)``.

    Returns
    -------
    np.ndarray
        Softmax-normalized probabilities of the same shape as input.
    """
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    normalizer = np.sum(exp_values, axis=1, keepdims=True)
    normalizer[normalizer == 0] = 1.0
    return exp_values / normalizer


def infer_frame_labels_and_probabilities(query_embeddings, positive_prototypes, background_prototypes):
    """Infer per-frame labels and foreground probability mass.

    Parameters
    ----------
    query_embeddings : np.ndarray
        Embeddings for the current audio file, shape
        ``(n_frames, embedding_dim)``.
    positive_prototypes : dict[str, np.ndarray]
        Positive class prototypes.
    background_prototypes : dict[str, np.ndarray]
        Background prototypes.

    Returns
    -------
    tuple[list[str], list[float], np.ndarray, list[str]]
        ``(frame_labels, positive_mass, distances, prototype_labels)`` where
        ``positive_mass`` is a single foreground trace produced by summing
        probabilities over non-background prototypes.
    """
    all_prototypes = {**positive_prototypes, **background_prototypes}
    if not all_prototypes:
        return [], [], [], []

    prototype_labels = list(all_prototypes.keys())
    prototype_matrix = np.stack([all_prototypes[label] for label in prototype_labels], axis=0)
    distances = np.linalg.norm(query_embeddings[:, None, :] - prototype_matrix[None, :, :], axis=2)
    nearest_indices = np.argmin(distances, axis=1)
    frame_labels = [prototype_labels[idx] for idx in nearest_indices]

    probabilities = softmax(-distances)
    positive_indices = [idx for idx, label in enumerate(prototype_labels) if not is_background_label(label)]

    if positive_indices:
        positive_mass = np.sum(probabilities[:, positive_indices], axis=1).tolist()
    else:
        positive_mass = [0.0] * len(frame_labels)

    return frame_labels, positive_mass, distances, prototype_labels