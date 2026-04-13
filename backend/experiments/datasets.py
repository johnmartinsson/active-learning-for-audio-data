"""Ground-truth dataset loading and management.

TODO
----
1. Define dataset interface/schema
   - Standardize: {embeddings, timings, labels, metadata}
   - Support multiple embedding models (BirdNET, custom, etc.)
   - Track ground truth label source/quality

2. Implement `load_groundtruth_dataset(dataset_name)` registry
   - Map dataset names to loader functions
   - Support arctic_foxes, guillemots, auklab, etc.
   - Return standardized dataset dict

3. Implement dataset loaders for each project dataset
   - `load_arctic_foxes()`
   - `load_guillemots()`
   - `load_auklab_clean_calls()`
   - etc.

4. Add data filtering/sampling utilities
   - Subset dataset by class, time range, etc.
   - Stratified sampling for balanced experiments
   - Train/val/test splits

5. Add dataset statistics/info functions
   - Duration, frame count, class distribution
   - Data quality checks (missing embeddings, malformed labels)

6. Support custom dataset registration
   - Allow users to add new datasets at runtime
   - Validation that custom datasets meet interface requirements
"""
