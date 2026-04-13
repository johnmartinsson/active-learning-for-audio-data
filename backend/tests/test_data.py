"""Unit tests for backend.python.acpd.data."""

import numpy as np

from python.acpd import data


def test_load_embeddings_and_timings_roundtrip(sample_timings, sample_embeddings, write_msgpack):
    msgpack_path = write_msgpack("embeddings/example.birdnet.embeddings.msgpack", sample_timings, sample_embeddings)
    timings, embeddings = data.load_embeddings_and_timings(str(msgpack_path))
    np.testing.assert_allclose(timings, sample_timings)
    np.testing.assert_allclose(embeddings, sample_embeddings)


def test_load_embeddings_and_timings_truncates_mismatched_lengths(sample_timings, sample_embeddings, write_msgpack):
    msgpack_path = write_msgpack(
        "embeddings/mismatch.birdnet.embeddings.msgpack",
        sample_timings[:2],
        sample_embeddings,
    )
    timings, embeddings = data.load_embeddings_and_timings(str(msgpack_path))
    assert len(timings) == 2
    assert len(embeddings) == 2


def test_to_centers_numpy_and_list_inputs(sample_timings):
    centers_np = data.to_centers(sample_timings)
    centers_list = data.to_centers(sample_timings.tolist())
    assert centers_np == [0.5, 1.5, 2.5, 3.5]
    assert centers_list == [0.5, 1.5, 2.5, 3.5]


def test_normalize_label_strips_and_lowercases():
    assert data.normalize_label(" Presence ") == "presence"
    assert data.normalize_label(123) == "123"
    assert data.normalize_label(None) == ""


def test_is_background_label_cases():
    assert data.is_background_label("background")
    assert data.is_background_label("absence")
    assert data.is_background_label("")
    assert data.is_background_label(" Background ")
    assert not data.is_background_label("class_label")


def test_normalize_segment_label_behavior():
    assert data.normalize_segment_label("background_0") == "background"
    assert data.normalize_segment_label("Class_Label") == "class_label"
    assert data.normalize_segment_label("") == "background"


def test_read_label_rows_header_and_malformed_lines(tmp_path):
    path = tmp_path / "labels" / "example.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "start_time,end_time,label\n"
        "0.0,1.0,background\n"
        "bad_line\n"
        "1.0,2.0,class_label\n",
        encoding="utf-8",
    )

    rows = data.read_label_rows(str(path))

    assert rows == [
        (0.0, 1.0, "background"),
        (1.0, 2.0, "class_label"),
    ]


def test_collect_labeled_embeddings_splits_positive_and_background(sample_dataset):
    class_embeddings, background_embeddings = data.collect_labeled_embeddings(
        str(sample_dataset["labels_dir"]),
        str(sample_dataset["embeddings_dir"]),
    )

    assert "presence" in class_embeddings
    assert len(class_embeddings["presence"]) == 3
    assert len(background_embeddings) == 1