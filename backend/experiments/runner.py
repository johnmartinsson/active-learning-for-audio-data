"""Experiment runner: Execute segmentation methods on datasets with given configurations.

TODO
----
1. Implement `run_method_on_dataset(method_name, dataset, config)` function
   - Load method from registry
   - Build context (embeddings, timings, labels)
   - Call method.run(context, config)
   - Return method result with metadata

2. Implement `run_experiment_grid(methods, datasets, configs)` function
   - Iterate over all combinations (methods × datasets × configs)
   - Run each combination
   - Collect results into structured format (e.g., pandas DataFrame)

3. Add result caching/memoization
   - Avoid re-running identical experiments
   - Store experiment hashes for reproducibility

4. Add logging/progress tracking
   - Print experiment status during batch runs
   - Record timing information per method/dataset

5. Handle failures gracefully
   - Continue on method errors (don't crash entire grid)
   - Log failures for later inspection
"""
