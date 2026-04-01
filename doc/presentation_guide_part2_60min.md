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
A2C introduces a value baseline `V(s)` and uses advantage:

```
Aₜ = Rₜ - V(sₜ)
```

This usually lowers gradient variance.

### Code map
- Algorithm: [benchmarks/a2c.py](../benchmarks/a2c.py)
- Standalone runner: [a2c_benchmark.py](../a2c_benchmark.py)

### What to point out in code
- Shared backbone + policy/value heads
- Policy loss + value loss + entropy term
- Episode-wise returns and advantage computation

### How A2C works step by step

```
1. Collect one full episode with current policy
2. Compute Monte Carlo discounted returns Gₜ for each step
3. Get V(sₜ) from critic head for each state
4. Compute advantage: Aₜ = Gₜ - V(sₜ)
5. Compute combined loss:
     L = -Σ log π(a|s) · A        (policy — weighted by advantage)
       + value_coef · (V(s) - G)²  (critic — learn to predict returns)
       - entropy_coef · H(π)       (entropy — prevent collapse)
6. Backpropagate and update both heads jointly
7. Repeat for many episodes
```

### Theory-to-code correspondence

| Theory concept | Code in `benchmarks/a2c.py` |
|---------------|----------------------------|
| `π(a\|s)` — actor | `ActorCritic.policy_head` → `Categorical(logits)` |
| `V(s)` — critic | `ActorCritic.value_head` → scalar |
| `A = G - V(s)` — advantage | `advantage = returns - values.detach()` |
| Policy loss | `-(log_probs * advantage).sum()` |
| Value loss | `F.mse_loss(values, returns)` |
| Entropy bonus | `dist.entropy().mean()` |
| Combined loss | `policy_loss + 0.5*value_loss - 0.01*entropy` |

### What A2C fixes vs REINFORCE
- **Lower variance** — advantage baseline removes noise from raw returns
- **Better credit assignment** — only above-average actions get reinforced
- **Exploration protection** — entropy bonus prevents premature collapse
- **Still on-policy** — each trajectory used once, then discarded

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

### Architecture in detail

```
┌─────────────┐     ┌─────────────┐
│  Worker 1   │     │  Worker 2   │    ... N workers
│  (own env)  │     │  (own env)  │
│  collect    │     │  collect    │
│  rollouts   │     │  rollouts   │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └──────┬────────────┘
              ▼
       ┌──────────────┐
       │ Shared Queue  │
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │   Learner     │
       │  apply grads  │
       │  update model │
       └──────────────┘
```

Each worker: runs its own env copy → collects `rollout_steps` transitions → computes advantages → pushes batch to queue.

Learner: dequeues batches → applies combined loss (same as A2C) → updates shared parameters.

Workers periodically refresh local weights from the shared model.

### Config defaults
`workers=4, rollout_steps=5, γ=0.99, lr=1e-3, value_coef=0.5, entropy_coef=0.01`

### A2C vs A3C
- **A2C** — synchronous updates, single process, easier to debug and reproduce
- **A3C** — async workers, multiprocessing, better wall-clock speed on multi-core CPUs
- **Trade-off**: multiprocessing adds engineering complexity (queues, sync, error handling)

---

## 5) PPO (Proximal Policy Optimization)

### Key intuition
PPO avoids overly large policy jumps with clipped objective:

```
L_clip(θ) = E[ min( rₜ(θ)·Aₜ , clip(rₜ(θ), 1-ε, 1+ε)·Aₜ ) ]

where rₜ(θ) = π_new(aₜ|sₜ) / π_old(aₜ|sₜ)
```

Often a practical default due to good stability/simplicity balance.

### Code map
- Algorithm: [benchmarks/ppo.py](../benchmarks/ppo.py)
- Standalone runner: [ppo_benchmark.py](../ppo_benchmark.py)

### What to point out in code
- Rollout buffer collection (1024 steps)
- GAE computation (`_compute_gae()`)
- Minibatch multi-epoch update with clipping

### The clipping mechanism explained

The probability ratio measures how much the policy changed:

```
r(θ) = π_new(a|s) / π_old(a|s)
```

- `r = 1.0` → policy unchanged
- `r = 1.5` → action 50% more likely under new policy
- `r = 0.5` → action 50% less likely

Clipping to `[1-ε, 1+ε]` (default `[0.8, 1.2]`) caps how much any single update can push:

```python
# benchmarks/ppo.py — the core clipping
ratio = torch.exp(new_log_probs - old_log_probs)
surr1 = ratio * advantage
surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantage
policy_loss = -torch.min(surr1, surr2).mean()
```

If advantage is positive (good action) and ratio > 1.2 → clip truncates the incentive to keep pushing.
If advantage is negative (bad action) and ratio < 0.8 → clip truncates the penalty.

### GAE (Generalized Advantage Estimation)

GAE blends multi-step TD errors with parameter `λ` (default 0.95):

```
λ=0:    pure 1-step TD (low variance, high bias)
λ=1:    full Monte Carlo (high variance, no bias)
λ=0.95: sweet spot — mostly Monte Carlo but smoothed by TD
```

```python
# benchmarks/ppo.py — _compute_gae()
delta = rewards[t] + gamma * next_v * not_done - values[t]
gae = delta + gamma * gae_lambda * not_done * gae
advantages[t] = gae
```

### Training flow
1. Collect rollout (1024 steps) with current policy, storing states, actions, log_probs, rewards, values
2. Compute advantages with GAE
3. Normalize advantages
4. Run 4 minibatch epochs (shuffle + split into batches of 64) on clipped loss
5. Discard rollout, repeat

### Config defaults
`rollout_steps=1024, update_epochs=4, minibatch_size=64, clip_eps=0.2, gae_lambda=0.95, lr=3e-4, value_coef=0.5, entropy_coef=0.01`

### Why PPO is the practical default
- **Stable** — clipping prevents catastrophic updates
- **Simple** — no constrained optimization (unlike TRPO)
- **Reuses data** — multiple epochs per rollout (more sample-efficient than A2C)
- **RLHF** — used to finetune ChatGPT, LLaMA, and other LLMs

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
- Dependency on `sb3-contrib`
- Callback-based reward capture
- Wrapper-based benchmark consistency with other methods

### The trust region constraint

```
maximize  E[ r(θ) · A ]
subject to  KL( π_old || π_new ) ≤ δ
```

KL divergence measures how different two probability distributions are. The constraint says: "improve the policy, but don't change it too much from the current one."

TRPO solves this exactly using conjugate gradient + line search. This is theoretically rigorous but computationally expensive per update.

### TRPO vs PPO
- **TRPO** — solves the constrained optimization exactly. Guaranteed monotonic improvement. Expensive.
- **PPO** — approximates the same idea with simple clipping. Much cheaper, nearly as stable, easier to implement.
- PPO was designed as a simpler alternative to TRPO. In practice, PPO is preferred for most tasks.

### When to use TRPO
- Safety-critical systems or robotics where conservative updates are essential
- When you need a theoretical monotonic improvement guarantee
- When compute budget allows the extra cost per update

### Code: benchmarks/trpo.py
This repo uses the `sb3-contrib` TRPO implementation wrapped for benchmark compatibility. Since sb3 manages the training loop internally, rewards are captured via callback for consistency with the other methods.

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