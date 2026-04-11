"""Scaffold tests for backend.python.acpd.data."""

from python.acpd import data


def test_load_embeddings_and_timings_smoke(sample_timings, sample_embeddings, write_msgpack, xfail_not_implemented):
    msgpack_path = write_msgpack("embeddings/example.birdnet.embeddings.msgpack", sample_timings, sample_embeddings)
    timings, embeddings = data.load_embeddings_and_timings(str(msgpack_path))
    assert len(timings) == len(embeddings)
    xfail_not_implemented("data.load_embeddings_and_timings")


def test_to_centers_smoke(sample_timings, xfail_not_implemented):
    centers = data.to_centers(sample_timings)
    assert len(centers) == len(sample_timings)
    xfail_not_implemented("data.to_centers")


def test_normalize_label_smoke(xfail_not_implemented):
    normalized = data.normalize_label(" Presence ")
    assert isinstance(normalized, str)
    xfail_not_implemented("data.normalize_label")


def test_is_background_label_smoke(xfail_not_implemented):
    is_background = data.is_background_label("background")
    assert isinstance(is_background, bool)
    xfail_not_implemented("data.is_background_label")


def test_normalize_segment_label_smoke(xfail_not_implemented):
    normalized = data.normalize_segment_label("background_0")
    assert isinstance(normalized, str)
    xfail_not_implemented("data.normalize_segment_label")


def test_read_label_rows_smoke(sample_label_file, xfail_not_implemented):
    label_path = sample_label_file()
    rows = data.read_label_rows(str(label_path))
    assert isinstance(rows, list)
    xfail_not_implemented("data.read_label_rows")


def test_collect_labeled_embeddings_smoke(sample_dataset, xfail_not_implemented):
    class_embeddings, background_embeddings = data.collect_labeled_embeddings(
        str(sample_dataset["labels_dir"]),
        str(sample_dataset["embeddings_dir"]),
    )
    assert isinstance(class_embeddings, dict)
    assert isinstance(background_embeddings, list)
    xfail_not_implemented("data.collect_labeled_embeddings")