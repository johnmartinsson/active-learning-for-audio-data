"""Scaffold tests for backend.python.acpd.prototypes."""

import numpy as np

from python.acpd import prototypes


def test_compute_mean_prototypes_single_vector():
    """Test that mean of single embedding equals that embedding."""
    class_embeddings = {"class_label": [np.array([0.0, 1.0])]}
    result = prototypes.compute_mean_prototypes(class_embeddings)
    assert "class_label" in result
    np.testing.assert_array_almost_equal(result["class_label"], np.array([0.0, 1.0]))


def test_compute_mean_prototypes_multiple_vectors():
    """Test mean computation over multiple embeddings per class."""
    class_embeddings = {
        "class_label": [np.array([0.0, 1.0]), np.array([2.0, 3.0])],
        "background": [np.array([1.0, 1.0]), np.array([3.0, 1.0])],
    }
    result = prototypes.compute_mean_prototypes(class_embeddings)
    assert len(result) == 2
    np.testing.assert_array_almost_equal(result["class_label"], np.array([1.0, 2.0]))
    np.testing.assert_array_almost_equal(result["background"], np.array([2.0, 1.0]))


def test_compute_mean_prototypes_empty_class():
    """Test that empty class lists are skipped."""
    class_embeddings = {"class_label": [np.array([0.0, 1.0])], "empty": []}
    result = prototypes.compute_mean_prototypes(class_embeddings)
    assert "class_label" in result
    assert "empty" not in result


def test_run_kmeans_empty():
    """Test k-means with empty data."""
    result = prototypes.run_kmeans(np.empty((0, 3)), k=2)
    assert result.shape == (0, 0)


def test_run_kmeans_single_point():
    """Test k-means with single point (should return that point)."""
    data = np.array([[1.0, 2.0, 3.0]])
    result = prototypes.run_kmeans(data, k=2)
    assert result.shape[0] >= 1
    assert result.shape[1] == 3


def test_run_kmeans_k_exceeds_samples():
    """Test k-means where k > n_samples (should clamp k)."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = prototypes.run_kmeans(data, k=5)
    assert result.shape[0] == 2  # Should clamp k to n_samples


def test_run_kmeans_convergence():
    """Test k-means converges and returns expected number of centroids."""
    # Create well-separated clusters
    data = np.vstack([
        np.random.default_rng(0).normal([0, 0], 0.1, (10, 2)),
        np.random.default_rng(1).normal([5, 5], 0.1, (10, 2)),
    ])
    result = prototypes.run_kmeans(data, k=2)
    assert result.shape == (2, 2)


def test_compute_background_prototypes_empty():
    """Test background prototypes with empty background data."""
    result = prototypes.compute_background_prototypes([], "none", 1)
    assert result == {}


def test_compute_background_prototypes_single_method():
    """Test single background prototype (method='none')."""
    bg_vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = prototypes.compute_background_prototypes(bg_vectors, "none", 2)
    assert "background" in result
    np.testing.assert_array_almost_equal(result["background"], np.array([2.5, 3.5, 4.5]))


def test_compute_background_prototypes_kmeans():
    """Test multiple background prototypes via k-means."""
    bg_vectors = [
        [0.0, 0.0, 0.0], [0.1, 0.1, 0.1],  # cluster 1
        [10.0, 10.0, 10.0], [10.1, 10.1, 10.1],  # cluster 2
    ]
    result = prototypes.compute_background_prototypes(bg_vectors, "kmeans", 2)
    assert len(result) == 2
    assert "background_0" in result
    assert "background_1" in result


def test_softmax_shape():
    """Test softmax preserves input shape."""
    scores = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]], dtype=float)
    result = prototypes.softmax(scores)
    assert result.shape == (2, 3)


def test_softmax_sums_to_one():
    """Test softmax rows sum to approximately 1.0."""
    scores = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]], dtype=float)
    result = prototypes.softmax(scores)
    row_sums = np.sum(result, axis=1)
    np.testing.assert_array_almost_equal(row_sums, np.ones(2))


def test_softmax_positive():
    """Test softmax outputs are all positive."""
    scores = np.array([[-1.0, 0.0, 1.0], [-10.0, -5.0, 0.0]], dtype=float)
    result = prototypes.softmax(scores)
    assert np.all(result > 0)


def test_softmax_numerical_stability():
    """Test softmax handles large values without overflow."""
    large_scores = np.array([[1e3, 1e3 + 1, 1e3 + 2]], dtype=float)
    result = prototypes.softmax(large_scores)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_infer_frame_labels_and_probabilities_empty():
    """Test with empty prototypes."""
    embeddings = np.array([[1.0, 2.0, 3.0]])
    frame_labels, positive_mass, probs, proto_labels = prototypes.infer_frame_labels_and_probabilities(
        embeddings, {}, {}
    )
    assert len(frame_labels) == 0
    assert len(positive_mass) == 0


def test_infer_frame_labels_and_probabilities_only_background():
    """Test when only background prototypes exist."""
    embeddings = np.array([[1.0, 2.0, 3.0]])
    bg_proto = {"background": np.array([1.0, 2.0, 3.0])}
    frame_labels, positive_mass, probs, proto_labels = prototypes.infer_frame_labels_and_probabilities(
        embeddings, {}, bg_proto
    )
    assert frame_labels == ["background"]
    assert positive_mass == [0.0]  # All background → no positive mass


def test_infer_frame_labels_and_probabilities_with_positive():
    """Test with both positive and background prototypes."""
    embeddings = np.array([
        [0.0, 0.0, 0.0],  # Close to positive
        [10.0, 10.0, 10.0],  # Close to background
    ])
    positive = {"class_label": np.array([0.0, 0.0, 0.0])}
    background = {"background": np.array([10.0, 10.0, 10.0])}
    frame_labels, positive_mass, probs, proto_labels = prototypes.infer_frame_labels_and_probabilities(
        embeddings, positive, background
    )
    assert len(frame_labels) == 2
    assert frame_labels[0] == "class_label"
    assert frame_labels[1] == "background"
    assert positive_mass[0] > positive_mass[1]  # First frame closer to positive


def test_infer_frame_labels_returns_four_values():
    """Test return tuple has exactly 4 elements."""
    embeddings = np.array([[1.0, 2.0]])
    positive = {"p": np.array([1.0, 2.0])}
    result = prototypes.infer_frame_labels_and_probabilities(embeddings, positive, {})
    assert len(result) == 4