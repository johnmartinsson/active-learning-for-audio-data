"""Scaffold tests for backend.python.acpd.prototypes."""

import numpy as np

from python.acpd import prototypes


def test_compute_mean_prototypes_smoke(xfail_not_implemented):
    class_embeddings = {"presence": [np.array([0.0, 1.0]), np.array([1.0, 2.0])]}
    result = prototypes.compute_mean_prototypes(class_embeddings)
    assert "presence" in result
    xfail_not_implemented("prototypes.compute_mean_prototypes")


def test_run_kmeans_smoke(sample_embeddings, xfail_not_implemented):
    centroids = prototypes.run_kmeans(sample_embeddings, k=2)
    assert centroids.shape[0] >= 1
    xfail_not_implemented("prototypes.run_kmeans")


def test_compute_background_prototypes_smoke(sample_embeddings, xfail_not_implemented):
    result = prototypes.compute_background_prototypes(sample_embeddings.tolist(), "kmeans", 2)
    assert isinstance(result, dict)
    xfail_not_implemented("prototypes.compute_background_prototypes")


def test_softmax_smoke(xfail_not_implemented):
    probabilities = prototypes.softmax(np.array([[0.1, 0.2, 0.3]], dtype=float))
    assert probabilities.shape == (1, 3)
    xfail_not_implemented("prototypes.softmax")


def test_infer_frame_labels_and_probabilities_smoke(sample_embeddings, xfail_not_implemented):
    positive_prototypes = {"presence": np.array([1.0, 1.0, 1.0])}
    background_prototypes = {"background": np.array([0.0, 0.0, 0.0])}
    result = prototypes.infer_frame_labels_and_probabilities(
        sample_embeddings,
        positive_prototypes,
        background_prototypes,
    )
    assert len(result) == 4
    xfail_not_implemented("prototypes.infer_frame_labels_and_probabilities")