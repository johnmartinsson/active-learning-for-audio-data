# Experiments Framework

This directory contains the research evaluation and experimentation layer for A-CPD method development.

## Overview

The experiments framework enables:
- **Method Development**: Develop new segmentation methods in `python/acpd/methods/` and evaluate them here
- **Dataset Management**: Load and manage ground-truth datasets uniformly
- **Systematic Evaluation**: Compare methods with standardized metrics
- **Active Learning Simulation**: Test iterative labeling scenarios
- **Hyperparameter Tuning**: Run grid searches over method configs

## Module Structure

- **`runner.py`**: Execute segmentation methods on datasets (batch processing)
- **`datasets.py`**: Load and manage ground-truth datasets uniformly
- **`evaluation.py`**: Compute metrics and compare methods
- **`configs.py`**: Define method configurations and hyperparameter grids
- **`results.py`**: Store, serialize, and analyze experiment results

## Typical Workflow

```python
# 1. Load a dataset
from experiments.datasets import load_groundtruth_dataset
dataset = load_groundtruth_dataset("arctic_foxes")

# 2. Define configs to test
from experiments.configs import create_grid
configs = create_grid("adaptive", {
    "change_point_prominence": [0.05, 0.1, 0.2],
    "embedding_window_size": [5, 10],
})

# 3. Run experiment grid
from experiments.runner import run_experiment_grid
results = run_experiment_grid(
    methods=["fixed", "adaptive"],
    datasets=[dataset],
    configs=configs,
)

# 4. Evaluate results
from experiments.evaluation import evaluate_method_on_dataset
metrics = evaluate_method_on_dataset(results, dataset["labels"])

# 5. Save results
from experiments.results import ExperimentResultSet
result_set = ExperimentResultSet(results, metrics)
result_set.save("experiment_2024_04_11.json")
```

## Scaffolding Status

Each module has detailed TODO lists. Development can proceed in any order, starting with whichever supports your immediate research needs.

Suggested starting points:
1. `datasets.py` - Load your real datasets
2. `runner.py` - Execute methods on those datasets  
3. `evaluation.py` - Measure performance
4. `configs.py` - Organize hyperparameter exploration
5. `results.py` - Track and compare experiments

## Integration with Production Code

The experiments layer is **loosely coupled** from the production pipeline:
- Methods (`python/acpd/methods/`) develop independently
- Registry (`python/acpd/registry.py`) is shared
- Pipeline (`python/acpd/pipeline.py`) orchestrates both paths

This ensures research code doesn't break production inference.
