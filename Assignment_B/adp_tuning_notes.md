# ADP tuning notes: training samples and online samples

These notes collect the two tuning points we want to try after the current ADP version is stable.

## 1. Increase offline training samples

Current configuration:

```python
N_STATE_SAMPLES = 120
N_TRAIN_NEXT_SAMPLES = 8
```

These parameters only affect offline training. They do not affect the runtime of the submitted policy after `ADP_THETA` has been fitted.

### What to try

Start with:

```python
N_STATE_SAMPLES = 250
N_TRAIN_NEXT_SAMPLES = 10
```

If training time is still acceptable, try:

```python
N_STATE_SAMPLES = 500
N_TRAIN_NEXT_SAMPLES = 10
```

### What to look at

After training, run:

```bash
python ADP_policy_27_with_checks_extract_fixed.py --check-values
python ADP_policy_27_with_checks_extract_fixed.py --smoke-test
```

Prefer configurations where:

- no Bellman targets fail,
- value predictions have sensible magnitudes,
- value predictions do not explode,
- mean predicted values generally decrease as the day progresses,
- the smoke test returns feasible actions without errors.

The final decision should be based on Task 6 simulation performance, not only the diagnostic output.

## 2. Tune online next-step samples

Current online policy configuration:

```python
N_NEXT_SAMPLES = 10
SOLVER_TIME_LIMIT = 4
```

These parameters affect the runtime of every `select_action(state)` call.

### What to try

If runtime is too slow:

```python
N_NEXT_SAMPLES = 5
```

If runtime is safe and the policy is unstable/noisy:

```python
N_NEXT_SAMPLES = 15
```

### What to measure

During Task 6 evaluation, store:

- average runtime per policy call,
- maximum runtime,
- number of calls close to or above the allowed runtime,
- total simulated daily cost,
- number of fallback actions, if tracked.

### Practical rule

Use the largest `N_NEXT_SAMPLES` that remains reliably below the evaluator's runtime limit. A slightly smaller sample size with no timeouts is usually better than a richer sample set that occasionally fails.
