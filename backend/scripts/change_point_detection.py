# backend/scripts/change_point_detection.py
import sys
import json
import numpy as np
from ruptures import Binseg
from scipy.signal import find_peaks

def detect_change_points_binary_segmentation(probabilities, num_segments):
    probabilities = np.array(probabilities)
    model = Binseg(model="l2").fit(probabilities)
    
    # Ensure the number of breakpoints is valid
    max_bkps = len(probabilities) - 1
    n_bkps = min(num_segments - 1, max_bkps)
    print("Number of breakpoints:", n_bkps, file=sys.stderr)
    print("Maximum number of breakpoints:", max_bkps, file=sys.stderr)
    
    if n_bkps <= 0:
        return []
    
    change_points = model.predict(n_bkps=n_bkps)
    return change_points

def euclidean_distance_score(x1, x2):
    return np.sqrt(np.sum(np.power(x1 - x2, 2)))

def distance_past_and_future_averages(embeddings, distance_fn, offset=5, M=5):
    ds = np.zeros(len(embeddings))

    for idx in range(M + offset, len(embeddings) - M - offset):
        past_start = idx - M - offset
        past_end = idx - offset
        future_start = idx + 1 + offset
        future_end = idx + M + 1 + offset

        past_mean = np.mean(embeddings[past_start:past_end, :], axis=0)
        future_mean = np.mean(embeddings[future_start:future_end, :], axis=0)

        distance = distance_fn(past_mean, future_mean)
        ds[idx] = distance

    return ds

def detect_change_points(probabilities, num_segments, prominence_threshold=0):
    probabilities = np.array(probabilities)
    probabilities = probabilities.reshape((len(probabilities), 1))
    ds = distance_past_and_future_averages(probabilities, euclidean_distance_score, offset=0, M=1)

    n_change_points = num_segments - 1

    peaks = find_peaks(ds, prominence=prominence_threshold)

    peak_indices = peaks[0]
    peak_prominences = peaks[1]['prominences']
    xs = sorted(list(zip(peak_indices, peak_prominences)), key=lambda x: x[1], reverse=True)

    top_n_peak_indices_sorted = sorted([int(x[0]) for x in xs[:n_change_points]])

    top_n_change_points = top_n_peak_indices_sorted
    return top_n_change_points

if __name__ == "__main__":
    input_data = json.loads(sys.stdin.read())
    # print("Received input data:", input_data, file=sys.stderr)  # Log input data to stderr
    probabilities = input_data["probabilities"]
    num_segments = input_data["num_segments"]
    prominence_threshold = input_data.get("prominence_threshold", 0)

    change_points = detect_change_points(probabilities, num_segments, prominence_threshold)
    print(json.dumps(change_points))
