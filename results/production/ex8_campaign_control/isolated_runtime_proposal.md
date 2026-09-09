# Proposed isolated-runtime protocol — NOT EXECUTED

The completed accuracy campaigns used four concurrent GPUs. Their timings are preserved, but shared-resource contention prevents direct publication comparisons with isolated benchmarks.

Propose 30 fresh runs per method, seeds 0–29, sequentially on physical GPU 0 with the entire machine otherwise idle. Use the frozen runners, data, arff-sde environment and normal per-process warm-up. Save to separate isolated-runtime directories; do not replace accuracy artifacts. Predeclare all seeds and report mean, sample SD, median and range of algorithm and warm-up time. Check all GPUs and CPU occupancy before every run; flag external interference without silently dropping results. Record GPU clocks/temperature and driver/runtime environment. Use a fixed, balanced alternating method order across seeds.

Baseline estimate from accepted isolated runs: approximately 5–6 hours including warm-up and artifact evaluation. Current parallel timing variability is recorded below for context, not as an isolated-runtime estimate. This protocol requires approval before execution.

```json
{
  "arff": {
    "new_parallel_runs": 29,
    "mean_algorithm_time": 7.745558781413851,
    "std_algorithm_time": 0.16993684466645106,
    "min_algorithm_time": 7.382961514987983,
    "max_algorithm_time": 8.24039749300573,
    "reused_seed": 1
  },
  "fourier": {
    "new_parallel_runs": 29,
    "mean_algorithm_time": 577.2712951054112,
    "std_algorithm_time": 18.475420763215283,
    "min_algorithm_time": 550.9709765099979,
    "max_algorithm_time": 627.7088555530063,
    "reused_seed": 0
  }
}
```
