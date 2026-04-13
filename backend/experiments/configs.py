"""Configuration management and hyperparameter grids for experiments.

TODO
----
1. Define config schema/validation
   - Validate config dicts before passing to methods
   - Type checking (int, float, str ranges)
   - Required vs. optional parameters per method

2. Create method-specific config templates
   - Fixed method config (minimal)
   - Adaptive method config (CPD parameters, window sizes, etc.)
   - Future method configs as they're added

3. Implement hyperparameter grid definitions
   - `create_grid(method_name, param_ranges)` → list of configs
   - Cartesian product of parameter values
   - Support logarithmic, linear, categorical ranges

4. Add config preset/named-config support
   - "default" config for each method
   - "aggressive" (more CPD peaks) vs. "conservative" (fewer peaks)
   - "fast" (small windows) vs. "accurate" (large windows)

5. Implement config versioning
   - Track config history for reproducibility
   - Config checksum for experiment matching

6. Add config loading from files (JSON/YAML)
   - Define experiments in config files
   - Print/save experiment configs for documentation
"""
