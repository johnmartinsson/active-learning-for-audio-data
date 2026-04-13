"""Evaluation metrics and comparison utilities for segmentation methods.

TODO
----
1. Implement segmentation quality metrics
   - Boundary F1: precision/recall of detected boundaries vs. ground truth
   - Intersection over Union (IoU) per segment
   - Hamming distance (pixel-level agreement)
   - Purity/homogeneity scores

2. Implement `evaluate_method_on_dataset(result, ground_truth_labels)` function
   - Take method output (segments, labels) and expected labels
   - Compute all metrics
   - Return metrics dict

3. Implement comparison utilities
   - `compare_methods(results_list, ground_truth)` → comparison table
   - Statistical significance testing (e.g., paired t-tests)
   - Visualization helpers (plots of method comparisons)

4. Add ablation study tools
   - Compare same method with different hyperparameters
   - Sensitivity analysis (which config params matter most?)

5. Implement per-class metrics
   - Performance breakdown by class (foxes, guillemots, etc.)
   - Class-specific strengths/weaknesses

6. Add active learning evaluation
   - Simulate iterative labeling scenario
   - Track how method uncertainty helps sampling strategy
   - Measure improvement curves over iterations
"""
