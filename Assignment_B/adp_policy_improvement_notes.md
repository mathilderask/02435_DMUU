# ADP Policy Improvement Notes

This note collects concrete improvement ideas for the Task 4 Approximate Dynamic Programming (ADP) policy. The goal is to keep the first implementation simple and defensible, while making it clear how we can extend it if performance is weak in the simulation environment.

The current ADP idea is:

```text
For the current state s_t:
    1. generate next-step uncertainty samples
    2. solve a one-step policy optimization problem
    3. minimize immediate cost + approximate future value
    4. return only the here-and-now action
```

Mathematically:

```text
min_a  c(s_t, a_t) + E[V_hat_{t+1}(s_{t+1})]
```

with a linear value function approximation:

```text
V_hat_t(s_t) = theta_t^T phi(s_t)
```

---

## 1. Use the full feature vector in the online optimization

### Motivation

The first ADP draft only uses a simple subset of the feature vector inside the Pyomo policy optimization. This is easy to implement, but it may miss important nonlinear risk indicators such as:

```text
humidity above threshold
temperature deficit below OK threshold
temperature above high threshold
```

The lecture discusses the use of features / basis functions. The value function remains linear in the coefficients theta, but the features phi(s) can contain nonlinear transformations of the state.

### Suggested feature vector

```python
def feature_vector(state):
    """
    Full ADP feature vector.

    The value function is linear in theta:
        V_hat_t(s) = theta_t @ phi(s)

    The features themselves may include nonlinear transformations such as
    positive parts. This keeps fitting linear while making the approximation
    more expressive.
    """
    T1 = state["T1"]
    T2 = state["T2"]
    H = state["H"]
    price = state["price_t"]
    price_prev = state["price_previous"]
    occ1 = state["Occ1"]
    occ2 = state["Occ2"]
    vent_counter = state["vent_counter"]
    low1 = state["low_override_r1"]
    low2 = state["low_override_r2"]

    return np.array([
        1.0,                              # constant
        price,                            # current price
        price_prev,                       # previous price
        occ1,                             # room 1 occupancy
        occ2,                             # room 2 occupancy
        H,                                # humidity level
        max(0.0, H - HHigh),              # humidity excess above threshold
        T1,                               # room 1 temperature
        T2,                               # room 2 temperature
        max(0.0, TOK - T1),               # room 1 temperature deficit
        max(0.0, TOK - T2),               # room 2 temperature deficit
        max(0.0, T1 - THigh),             # room 1 high-temperature excess
        max(0.0, T2 - THigh),             # room 2 high-temperature excess
        vent_counter,                     # ventilation inertia state
        low1,                             # low-temperature overrule room 1
        low2,                             # low-temperature overrule room 2
    ], dtype=float)
```

### Pyomo implementation for positive-part features

If the online ADP optimization uses these features for the next state, positive-part terms need auxiliary variables.

```python
# Auxiliary variables for nonlinear but piecewise-linear features
m.h_excess = pyo.Var(m.K, bounds=(0, None))
m.t1_deficit = pyo.Var(m.K, bounds=(0, None))
m.t2_deficit = pyo.Var(m.K, bounds=(0, None))
m.t1_excess = pyo.Var(m.K, bounds=(0, None))
m.t2_excess = pyo.Var(m.K, bounds=(0, None))

for k in m.K:
    # max(0, H_next - HHigh)
    m.cons.add(m.h_excess[k] >= m.H_next[k] - HHigh)

    # max(0, TOK - T_next)
    m.cons.add(m.t1_deficit[k] >= TOK - m.T1_next[k])
    m.cons.add(m.t2_deficit[k] >= TOK - m.T2_next[k])

    # max(0, T_next - THigh)
    m.cons.add(m.t1_excess[k] >= m.T1_next[k] - THigh)
    m.cons.add(m.t2_excess[k] >= m.T2_next[k] - THigh)
```

Then include these terms in the approximate future value:

```python
future_value = 0.0

for k, (price_next, occ1_next, occ2_next) in enumerate(samples):
    future_value += prob * (
        theta_next[0]
        + theta_next[1] * price_next
        + theta_next[2] * price
        + theta_next[3] * occ1_next
        + theta_next[4] * occ2_next
        + theta_next[5] * m.H_next[k]
        + theta_next[6] * m.h_excess[k]
        + theta_next[7] * m.T1_next[k]
        + theta_next[8] * m.T2_next[k]
        + theta_next[9] * m.t1_deficit[k]
        + theta_next[10] * m.t2_deficit[k]
        + theta_next[11] * m.t1_excess[k]
        + theta_next[12] * m.t2_excess[k]
        # optional: add next ventilation counter / low-overrule features
        # if these are explicitly modeled in the one-step problem
    )
```

### Priority

High. This is probably the most useful improvement because it directly improves how the ADP policy values future risk.

---

## 2. Use stage-dependent value functions

### Motivation

The value of a state depends strongly on the remaining number of hours. A cold room early in the day matters more than a cold room near closing time. Therefore, a separate value function per hour is better than one global value function.

Use:

```text
V_hat_t(s_t) = theta_t^T phi(s_t)
```

instead of:

```text
V_hat(s_t) = theta^T phi(s_t)
```

### Implementation

```python
# Stage-dependent coefficients
ADP_THETA = {
    0: [...],
    1: [...],
    2: [...],
    3: [...],
    4: [...],
    5: [...],
    6: [...],
    7: [...],
    8: [...],
    9: [...],
}

current_time = int(round(safe_float(state.get("current_time", 0), 0)))
theta_next = ADP_THETA.get(current_time + 1, [0.0] * N_FEATURES)
```

### Priority

High. This should be part of the base implementation.

---

## 3. Improve training samples using forward rollouts

### Motivation

Purely random training states may be unrealistic. The value function should be accurate in the regions of the state space that the policy actually visits.

The lecture notes mention the issue that random samples can be inconsistent across stages. A forward-backward algorithm addresses this by first simulating the policy forward and then fitting the value function on visited states.

### Algorithm sketch

```text
Initialize theta_t
For iteration = 1, ..., I:
    Forward pass:
        simulate many days using current ADP policy
        store visited states by time step

    Backward pass:
        for t = T-1, ..., 0:
            compute Bellman targets for stored states at time t
            refit theta_t by regression
```

### Code skeleton

```python
def collect_rollout_states(policy, n_days):
    """
    Simulate the current policy and collect states actually visited.
    These states are later used as training samples for the value function.
    """
    states_by_time = {t: [] for t in range(T_HORIZON)}

    for d in range(n_days):
        state = initial_state_for_training_day(d)

        for t in range(T_HORIZON):
            states_by_time[t].append(state.copy())

            action = policy.select_action(state)
            effective_action = apply_overrules_in_environment(state, action)
            state = environment_step(state, effective_action)

    return states_by_time


def forward_backward_training(n_iterations):
    theta = initialize_theta()

    for it in range(n_iterations):
        policy = build_adp_policy_from_theta(theta)

        # Forward pass: sample states where the policy actually goes
        states_by_time = collect_rollout_states(policy, n_days=100)

        # Backward pass: refit value functions
        for t in reversed(range(T_HORIZON)):
            targets = []
            states = states_by_time[t]

            for s in states:
                target = solve_one_step_bellman_target(
                    state=s,
                    theta_next=theta[t + 1],
                )
                targets.append(target)

            theta[t] = fit_stage_theta(states, targets)

    return theta
```

### Priority

Medium to high. This is a strong improvement once the basic ADP is running.

---

## 4. Use approximate policy iteration

### Motivation

In a simple backward pass, the policy can change while the value function is still noisy. Approximate policy iteration separates:

1. policy evaluation: estimate the value of the current policy,
2. policy improvement: update the policy after evaluation.

This is more stable, but also more implementation work.

### Algorithm sketch

```text
Initialize policy pi_0 or theta_0

For outer iteration i:
    Policy evaluation:
        keep policy pi_i fixed
        estimate/fix theta_i using several evaluation sweeps

    Policy improvement:
        define pi_{i+1} using the improved value function theta_i
```

### Code skeleton

```python
def approximate_policy_iteration(n_outer, n_eval_sweeps):
    theta = initialize_theta()

    for i in range(n_outer):
        # Current policy is fixed during the evaluation step
        fixed_policy = build_adp_policy_from_theta(theta)

        for sweep in range(n_eval_sweeps):
            states_by_time = collect_rollout_states(fixed_policy, n_days=100)

            for t in reversed(range(T_HORIZON)):
                targets = []

                for s in states_by_time[t]:
                    # Important: use the fixed policy or fixed actions
                    # to evaluate the current policy, rather than improving
                    # the decision at the same time.
                    target = evaluate_fixed_policy_target(
                        state=s,
                        policy=fixed_policy,
                        theta_next=theta[t + 1],
                    )
                    targets.append(target)

                theta[t] = fit_stage_theta(states_by_time[t], targets)

        # Policy improvement happens after evaluation
        improved_policy = build_adp_policy_from_theta(theta)

    return theta
```

### Priority

Medium. Good for report discussion or later improvement, but not necessary for the first working version.

---

## 5. Use Optimal-in-Hindsight fitted values as warm start

### Motivation

The lecture mentions using an Optimal-in-Hindsight oracle to generate initial value estimates. This can produce a useful initial value function before running fitted value iteration or backward induction.

For this assignment, we already have an OiH MILP from Part A. We can reuse it as an oracle.

### Algorithm sketch

```text
For each time step t:
    sample states s_t
    sample future trajectories from t to end of day
    solve the OiH MILP for the remaining horizon
    use the optimal remaining cost as target value
    fit theta_t to these targets
```

### Code skeleton

```python
def train_oih_warm_start():
    theta = {}

    for t in reversed(range(T_HORIZON)):
        states_t = sample_training_states(t, n_samples=100)
        targets_t = []

        for s in states_t:
            # Generate a full future trajectory of prices and occupancies
            trajectory = sample_future_trajectory_from_state(s, start_time=t)

            # Solve OiH from this state with the sampled future known
            target = solve_remaining_horizon_oih(
                initial_state=s,
                trajectory=trajectory,
                start_time=t,
            )

            targets_t.append(target)

        theta[t] = fit_stage_theta(states_t, targets_t)

    return theta
```

### Possible report wording

```text
As an initialization, we fitted the value function to targets produced by the
Optimal-in-Hindsight model on sampled future trajectories. This gave a stable
initial estimate of the remaining-day cost-to-go, which was then optionally
refined with approximate backward induction.
```

### Priority

Medium. Strong academically, but potentially expensive offline.

---

## 6. Piecewise value functions for binary state variables

### Motivation

The lecture mentions piecewise linear value-function approximation for binary endogenous states. Our state contains binary / discrete controller states:

```text
low_override_r1
low_override_r2
vent_counter
```

The value of a state can differ significantly depending on whether a low-temperature overrule is active. Therefore, instead of one value function, we can fit separate value functions for different controller modes.

### Example

Fit four value functions per time step, one for each low-overrule combination:

```text
V_hat_t^{0,0}(s)
V_hat_t^{1,0}(s)
V_hat_t^{0,1}(s)
V_hat_t^{1,1}(s)
```

where the superscript denotes:

```text
(low_override_r1, low_override_r2)
```

### Code skeleton

```python
ADP_THETA = {
    # t: {
    #     (low1, low2): theta_vector
    # }
    0: {
        (0, 0): [...],
        (1, 0): [...],
        (0, 1): [...],
        (1, 1): [...],
    },
    1: {
        (0, 0): [...],
        (1, 0): [...],
        (0, 1): [...],
        (1, 1): [...],
    },
    # ...
}


def select_theta_for_state(state, current_time):
    low1 = int(state.get("low_override_r1", 0) > 0.5)
    low2 = int(state.get("low_override_r2", 0) > 0.5)

    return ADP_THETA[current_time][(low1, low2)]
```

For training:

```python
def fit_piecewise_theta(states, targets):
    theta_by_mode = {}

    for low1 in [0, 1]:
        for low2 in [0, 1]:
            mode_states = [
                s for s in states
                if int(s["low_override_r1"]) == low1
                and int(s["low_override_r2"]) == low2
            ]

            mode_targets = [
                y for s, y in zip(states, targets)
                if int(s["low_override_r1"]) == low1
                and int(s["low_override_r2"]) == low2
            ]

            if len(mode_states) >= MIN_SAMPLES_PER_MODE:
                theta_by_mode[(low1, low2)] = fit_stage_theta(
                    mode_states,
                    mode_targets,
                )
            else:
                # fallback to global theta if too few samples
                theta_by_mode[(low1, low2)] = fit_stage_theta(states, targets)

    return theta_by_mode
```

### Priority

Medium. Potentially useful, but more complex. Good optional improvement if the global linear value function is too rough.

---

## 7. Add regularization in the regression

### Motivation

The value-function fit may become unstable if features are correlated. Ridge regression is a simple and robust fix.

### Implementation

```python
def fit_stage_theta(states, targets, ridge=1e-5):
    Phi = np.vstack([feature_vector(s) for s in states])
    y = np.array(targets, dtype=float)

    A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    b = Phi.T @ y

    theta = np.linalg.solve(A, b)
    return theta
```

### Priority

High. This should be included from the beginning.

---

## 8. Standardize features before fitting

### Motivation

Features have different scales:

```text
temperature around 16-26
humidity around 35-80
price around 0-12
occupancy around 10-50
binary states around 0-1
```

Without scaling, the regression can become numerically unstable and coefficients are harder to interpret.

### Implementation idea

```python
def fit_scaler(states):
    Phi = np.vstack([raw_feature_vector(s) for s in states])
    mean = Phi.mean(axis=0)
    std = Phi.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def scaled_features(state, mean, std):
    return (raw_feature_vector(state) - mean) / std
```

If using scaled features, the submitted policy must include the same means and standard deviations.

### Priority

Medium. Useful if regression becomes unstable.

---

## 9. Use several next-step samples in the online ADP optimization

### Motivation

Using a single expected transition ignores uncertainty. ADP should ideally minimize:

```text
immediate cost + expected approximate future value
```

So the online optimization should average over several next-step samples.

### Implementation

```python
samples = []
for _ in range(N_NEXT_SAMPLES):
    next_price = price_model(price, price_prev)
    next_occ1, next_occ2 = next_occupancy_levels(occ1, occ2)
    samples.append((next_price, next_occ1, next_occ2))

prob = 1.0 / N_NEXT_SAMPLES

future_value = sum(
    prob * V_hat_next(next_state_for_sample[k])
    for k in range(N_NEXT_SAMPLES)
)
```

### Priority

High. This is already in the first trained ADP draft and should remain.

---

## 10. Keep the online policy small and fast

### Motivation

The submitted policy is called at every hour, and if it is too slow the dummy action is used. Therefore, online ADP should be much lighter than the multi-stage SP policy.

Recommended online settings:

```python
N_NEXT_SAMPLES = 5 to 15
SOLVER_TIME_LIMIT = 3 to 5 seconds
```

If using a candidate-action enumeration instead of Pyomo:

```python
heat_candidates = [0, 0.25 * Pmax, 0.5 * Pmax, 0.75 * Pmax, Pmax]
ventilation_candidates = [0, 1]
```

This gives only 50 candidate actions and is very fast.

### Priority

High. Robustness matters more than a theoretically perfect but slow policy.

---

# Recommended implementation order

## Base version

1. Stage-dependent theta_t.
2. Ridge regression training.
3. One-step online ADP optimization.
4. Several next-step samples.
5. Low-temperature overrule state included.

## First improvement

6. Full feature vector with positive-part features.
7. Auxiliary Pyomo variables for humidity excess and temperature deficits.
8. Evaluate in Task 6 environment and compare to dummy / deterministic lookahead / SP.

## Second improvement

9. Generate training states from forward rollouts instead of purely random sampling.
10. Refit theta_t using backward induction on visited states.

## Optional advanced improvements

11. OiH-fitted warm start.
12. Piecewise value functions for low-overrule modes.
13. Approximate policy iteration.

---

# Suggested report wording

```text
The ADP policy uses a stage-dependent linear value function approximation
V_hat_t(s_t) = theta_t^T phi(s_t). The feature vector includes the current
thermal and humidity state, price and occupancy information, ventilation
inertia, low-temperature overrule states, and positive-part features measuring
humidity excess and temperature deficits. The coefficients theta_t are fitted
offline by sampling-based approximate backward induction. For each sampled
state, several next-step uncertainty samples are generated and a one-step
Bellman optimization problem is solved to compute a target value. The
coefficients are then obtained by ridge regression.

Online, the policy observes the current state, generates next-step uncertainty
samples, and solves a one-step optimization problem minimizing the immediate
electricity cost plus the expected approximate value of the next state. Only
the resulting here-and-now heating and ventilation decisions are returned to
the environment.
```
