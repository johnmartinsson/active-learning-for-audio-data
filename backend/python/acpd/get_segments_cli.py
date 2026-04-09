import json
import os
import sys

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


def to_centers(timings):
    return [float((start + end) / 2.0) for start, end in timings]


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

    # First migration step: prove the Node -> Python -> JSON roundtrip works.
    # For the "active" strategy we intentionally return a simple 5-segment layout.
    num_segments = 5 if labeling_strategy_choice == "active" else requested_num_segments

    timings = []
    timings_centers = []
    source = "fixed-fallback"

    if embeddings_path and os.path.exists(embeddings_path):
        timings = load_timings(embeddings_path)
        timings_centers = to_centers(timings)
        segments = build_timing_aware_segments(audio_length, num_segments, timings)
        source = "msgpack-timings"
    else:
        segments = build_fixed_segments(audio_length, num_segments)

    response = {
        "segments": segments,
        "probabilities": [],
        "timings": timings_centers,
        "suggestedLabels": ["background"] * len(segments),
        "backend": {
            "engine": "python",
            "strategy": labeling_strategy_choice,
            "requested_num_segments": requested_num_segments,
            "returned_num_segments": len(segments),
            "module": "python.acpd.get_segments_cli",
            "source": source,
        },
    }

    print(json.dumps(response))


if __name__ == "__main__":
    main()
