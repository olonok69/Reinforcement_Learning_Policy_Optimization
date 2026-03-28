# RL Policy Optimization — Presentation Guide Part 2 (60 minutes)

## Scope of Part 2
This session covers the remaining policy optimization algorithms after REINFORCE:
- A2C
- A3C
- PPO
- TRPO

Focus:
- Why these methods improve stability/efficiency over vanilla REINFORCE
- How each method maps to repository code
- How to compare methods with reproducible benchmark workflow

---

## 1) Session objective
By the end of this Part 2 talk, the audience should understand:
- The role of critic baselines in reducing variance
- Why asynchronous rollout collection can help (A3C)
- Why clipped/trust-region updates stabilize policy learning (PPO/TRPO)
- How to run and compare all methods in this repository

---

## 2) Suggested 60-minute agenda

- **0–6 min**: quick recap from Part 1 (REINFORCE limitations)
- **6–20 min**: A2C (actor + critic)
- **20–30 min**: A3C (async workers)
- **30–43 min**: PPO (clipped surrogate + GAE)
- **43–50 min**: TRPO (trust-region concept + library integration)
- **50–57 min**: benchmark execution/comparison workflow
- **57–60 min**: recommendation summary + Q&A

---

## 3) A2C (Advantage Actor-Critic)

### Key intuition
REINFORCE uses full returns directly and can be noisy.
A2C introduces a value baseline $V(s)$ and uses advantage:

$$
\hat{A}_t = R_t - V(s_t)
$$

This usually lowers gradient variance.

### Code map
- Algorithm: [benchmarks/a2c.py](../benchmarks/a2c.py)
- Standalone runner: [a2c_benchmark.py](../a2c_benchmark.py)

### What to point out in code
- Shared backbone + policy/value heads
- Policy loss + value loss + entropy term
- Episode-wise returns and advantage computation

---

## 4) A3C (Asynchronous Advantage Actor-Critic)

### Key intuition
A3C keeps actor-critic structure but collects experience using multiple workers in parallel.

Benefits:
- data decorrelation
- wall-clock speedup on CPU

Trade-off:
- more systems complexity (processes, queues, synchronization)

### Code map
- Algorithm: [benchmarks/a3c.py](../benchmarks/a3c.py)
- Standalone runner: [a3c_benchmark.py](../a3c_benchmark.py)

### What to point out in code
- Worker loop collecting episodes and mini-rollouts
- Learner consuming batch queue
- Shared parameter refresh pattern

---

## 5) PPO (Proximal Policy Optimization)

### Key intuition
PPO avoids overly large policy jumps with clipped objective:

$$
L^{clip}(\theta)=\mathbb{E}[\min(r_t(\theta)\hat{A}_t,\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)]
$$

Often a practical default due to good stability/simplicity balance.

### Code map
- Algorithm: [benchmarks/ppo.py](../benchmarks/ppo.py)
- Standalone runner: [ppo_benchmark.py](../ppo_benchmark.py)

### What to point out in code
- rollout buffer collection
- GAE computation
- minibatch multi-epoch update with clipping

---

## 6) TRPO (Trust Region Policy Optimization)

### Key intuition
TRPO constrains policy updates to remain in a trust region (typically KL-constrained).

Pros:
- conservative, principled updates

Cons:
- heavier optimization machinery

### Code map
- Integration wrapper: [benchmarks/trpo.py](../benchmarks/trpo.py)
- Standalone runner: [trpo_benchmark.py](../trpo_benchmark.py)

### What to point out in code
- dependency on `sb3-contrib`
- callback-based reward capture
- wrapper-based benchmark consistency with other methods

---

## 7) Side-by-side comparison narrative

| Method | Main idea | Strength | Common trade-off |
|---|---|---|---|
| A2C | Critic baseline + advantage | More stable than REINFORCE | Still on-policy and sample-hungry |
| A3C | Async multi-worker actor-critic | Better wall-clock behavior on CPU | Multiprocessing complexity |
| PPO | Clipped objective + repeated minibatch updates | Strong practical default | Hyperparameter-sensitive |
| TRPO | Trust-region constrained updates | Stable conservative improvement | Computationally heavier |

---

## 8) Demo commands (Part 2)

### Run each method
```bash
uv run python a2c_benchmark.py
uv run python a3c_benchmark.py
uv run python ppo_benchmark.py
uv run python trpo_benchmark.py
```

### Run all methods with one command
```bash
uv run python run_all_comparison.py --methods policy_gradient a2c a3c ppo trpo
```

### Run selected methods with videos
```bash
uv run python run_all_comparison.py --methods a2c ppo trpo --record-video --video-dir videos --video-episodes 2
```

### Example budget alignment override
```bash
uv run python run_all_comparison.py --methods a2c a3c ppo trpo --a2c-episodes 600 --a3c-episodes 600 --ppo-episodes 700 --trpo-timesteps 200000
```

---

## 9) Reporting and interpretation

Primary metrics produced by comparison runner:
- `max_avg_reward_100`
- `final_avg_reward_100`
- `elapsed_sec`
- `episodes`

Recommended interpretation:
- Compare peak capability (`max_avg_reward_100`)
- Compare end stability (`final_avg_reward_100`)
- Compare practical cost (`elapsed_sec`)
- Repeat across multiple seeds and aggregate

Aggregation pipeline:
- [../scripts/aggregate_results.py](../scripts/aggregate_results.py)
- [../scripts/generate_aggregate_report.py](../scripts/generate_aggregate_report.py)

---

## 10) Close and recommendations

Suggested practical baseline order:
1. Start with PPO.
2. Use A2C for simpler actor-critic baseline and faster iteration.
3. Use A3C when CPU parallel collection is desirable.
4. Use TRPO when conservative policy updates are required and extra compute is acceptable.

For foundational intuition, always anchor audience in Part 1 (policy optimization + Monte Carlo + REINFORCE) before deep-diving these variants.
