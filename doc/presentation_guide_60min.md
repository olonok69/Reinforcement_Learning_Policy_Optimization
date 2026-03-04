# RL Policy Optimization Benchmarks — 60-Minute Presentation Guide

## 1) Session objective
By the end of this talk, the audience should understand:
- What this application does
- Core RL concepts (MDP, return, exploration/exploitation)
- What **Policy Optimization** means in practice
- How each implemented algorithm maps to code in this repository
- How to run and compare all methods end-to-end

---

## 2) Suggested 60-minute agenda

- **0–5 min**: Problem framing and application overview
- **5–15 min**: RL fundamentals (MDP, return, exploration/exploitation)
- **15–25 min**: Policy objective and policy gradient intuition
- **25–40 min**: Algorithms implemented (code-linked walkthrough)
- **40–50 min**: Live run flow (single method → all methods → aggregate report)
- **50–57 min**: Results interpretation and trade-offs
- **57–60 min**: Key takeaways and Q&A

---

## 3) What this application is

This repository is a benchmarking application for policy-optimization RL algorithms on `CartPole-v1`.

### Core capabilities
- Run each algorithm independently (one script per method)
- Run all algorithms from one orchestrator
- Save standardized metrics (`episodes`, `elapsed_sec`, `max_avg_reward_100`, `final_avg_reward_100`)
- Aggregate multiple runs/seeds
- Generate plots and a Markdown report

### Entry points
- Unified orchestrator: [run_all_comparison.py](../run_all_comparison.py)
- Compatibility entrypoint: [rl_comparison.py](../rl_comparison.py)

### Shared benchmarking utilities
- [benchmarks/common.py](../benchmarks/common.py)

---

## 4) RL concepts you should explain

### 4.1 MDP framing
An RL problem is usually modeled as a Markov Decision Process (MDP):
- state $s_t$
- action $a_t$
- reward $r_t$
- transition dynamics $P(s_{t+1}|s_t,a_t)$
- discount factor $\gamma$

The objective is to maximize expected discounted return:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

### 4.2 What is Policy Optimization?
Policy Optimization methods directly optimize policy parameters $\theta$ of $\pi_\theta(a|s)$ to maximize expected return:

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

Typical gradient idea:

$$
\nabla_\theta J(\theta) \propto \mathbb{E}[\nabla_\theta\log \pi_\theta(a_t|s_t)\,\hat{A}_t]
$$

Where $\hat{A}_t$ (advantage) tells whether an action was better or worse than baseline expectation.

### 4.3 Why optimize the policy directly?
Policy Optimization is useful when we care about learning a robust action distribution, not only a value estimate.

Key intuition:
- A policy is a probability distribution over actions.
- If an action leads to better-than-expected outcomes ($\hat{A}_t > 0$), increase its probability.
- If an action leads to worse-than-expected outcomes ($\hat{A}_t < 0$), decrease its probability.

The gradient term $\nabla_\theta \log \pi_\theta(a_t|s_t)$ points in the direction that increases action likelihood. Multiplying by $\hat{A}_t$ turns this into a reward-weighted update.

### 4.4 Variance, bias, and stability in practice
In real training, policy gradients can be noisy. Most algorithms in this repo address that with:
- **Baselines / critics** (A2C, A3C, PPO, TRPO) to reduce variance.
- **Entropy regularization** to prevent premature deterministic policies.
- **Trust constraints or clipping** (TRPO/PPO) to avoid destructive parameter jumps.

### 4.5 On-policy workflow (common to these methods)
The implemented methods mainly follow an on-policy loop:
1. Roll out trajectories with current policy.
2. Estimate returns/advantages.
3. Optimize policy (and value model when present).
4. Discard old trajectories and repeat with updated policy.

This is generally stable, but can be less sample-efficient than replay-buffer methods.

---

## 5) Algorithm map (with code links)

### Policy Gradient (REINFORCE)
- Code: [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)
- Runner: [policy_gradient_benchmark.py](../policy_gradient_benchmark.py)
- Key point: Monte Carlo returns directly weight policy log-prob gradients.
- Presenter note: emphasize this as the "pure" estimator; simple but high-variance.

### A2C (Advantage Actor-Critic)
- Code: [benchmarks/a2c.py](../benchmarks/a2c.py)
- Runner: [a2c_benchmark.py](../a2c_benchmark.py)
- Key point: actor + critic; advantage reduces variance vs plain REINFORCE.
- Presenter note: actor decides, critic explains; this decomposition is usually the first major stability jump.

### A3C (Asynchronous Advantage Actor-Critic)
- Code: [benchmarks/a3c.py](../benchmarks/a3c.py)
- Runner: [a3c_benchmark.py](../a3c_benchmark.py)
- Key point: multi-process workers collect rollouts asynchronously; central learner updates shared model.
- Presenter note: highlight wall-clock speedups and decorrelated data from parallel workers.

### PPO (Proximal Policy Optimization)
- Code: [benchmarks/ppo.py](../benchmarks/ppo.py)
- Runner: [ppo_benchmark.py](../ppo_benchmark.py)
- Key point: clipped objective for stable policy improvement.
- Presenter note: this is often the practical default due to simplicity/stability balance.

### TRPO (Trust Region Policy Optimization)
- Code: [benchmarks/trpo.py](../benchmarks/trpo.py)
- Runner: [trpo_benchmark.py](../trpo_benchmark.py)
- Key point: trust-region constrained update (implemented via `sb3-contrib` integration).
- Presenter note: conceptually principled and conservative updates, usually with heavier optimization cost.

---

## 6) How to run (demo-ready commands)

### 6.1 Environment setup
```bash
uv sync
```

### 6.2 Run one algorithm
```bash
uv run python ppo_benchmark.py
```

### 6.3 Run all algorithms
```bash
uv run python run_all_comparison.py
```

Or selected subset:
```bash
uv run python run_all_comparison.py --methods policy_gradient ppo trpo
```

### 6.4 Aggregate multi-seed outputs
```bash
uv run python scripts/aggregate_results.py --inputs outputs/*.json --output-json outputs/aggregate_summary.json --output-csv outputs/aggregate_summary.csv
```

### 6.5 Generate plots + report
```bash
uv run python scripts/generate_aggregate_report.py --input outputs/aggregate_summary.json --output-dir outputs/report --title "RL Policy Optimization Aggregate Report"
```

Generated report:
- [outputs/report/aggregate_report.md](../outputs/report/aggregate_report.md)

---

## 7) How to present the output metrics

Primary metrics in this app:
- `max_avg_reward_100`: best moving average over 100 episodes (peak capability)
- `final_avg_reward_100`: average over last 100 episodes (end-of-training stability)
- `elapsed_sec`: wall-clock time (compute efficiency)
- `episodes`: total episodes used

Interpretation recommendation:
- Compare **peak** (`max_avg_reward_100`) for best discovered behavior
- Compare **final** (`final_avg_reward_100`) for convergence quality
- Compare **time** for practical cost
- Use multiple seeds for statistical confidence

---

## 8) Suggested slide narrative (concise)

1. **Why RL benchmark app?**
   - Need apples-to-apples comparison under shared environment and metrics.

2. **Policy optimization intuition**
   - Directly improve policy quality from returns and advantage estimates.
   - Explain why clipped/trust-region updates reduce catastrophic policy shifts.

3. **Algorithm progression**
   - REINFORCE → A2C/A3C → PPO/TRPO.

4. **Engineering architecture**
   - Independent scripts + one orchestrator + aggregation + report generation.

5. **Results and trade-offs**
   - Performance, stability, compute time.

6. **Practical recommendation**
   - For production-like baselines: PPO/A2C, with TRPO for constrained-update scenarios.

---

## 9) Quick Q&A prep

- **Why does one algorithm score higher but take much longer?**
  - Different sample efficiency, optimization style, and compute overhead.

- **Why do we need multiple seeds?**
  - RL has high variance; one run can be misleading.

- **Why keep multiple policy optimization methods?**
  - They provide different stability/efficiency trade-offs and practical deployment options.

---

## 10) 5-minute whiteboard script

Use this as a fast spoken walkthrough:

1. **Goal (30s)**
   - "We want a policy that maps states to actions maximizing long-term reward."

2. **Core object (45s)**
   - "Our policy is $\pi_\theta(a|s)$, a probability distribution over actions."
   - "Training means adjusting $\theta$ so better actions become more likely."

3. **Learning signal (60s)**
   - "After rollouts, we estimate advantage $\hat{A}_t$."
   - "If $\hat{A}_t>0$, increase probability of that action; if $\hat{A}_t<0$, decrease it."
   - "This is captured by $\nabla_\theta \log \pi_\theta(a_t|s_t)\,\hat{A}_t$."

4. **Algorithm progression (90s)**
   - "REINFORCE: pure Monte Carlo policy gradient, simple but noisy."
   - "A2C/A3C: add critic baseline to reduce variance; A3C adds async workers."
   - "PPO/TRPO: constrain update size (clip or trust region) for stable improvement."

5. **Practical trade-off (45s)**
   - "More conservative updates often improve stability but can cost extra compute/time."
   - "PPO is typically the practical default; TRPO for stricter policy-step control."

6. **Close (30s)**
   - "We compare methods with shared metrics: peak reward, final reward, and elapsed time."
   - "Always validate across multiple seeds due to RL variance."

---

## 11) Additional repo docs
- Main project guide: [README.md](../README.md)
- Algorithm-specific docs index: [doc/README.md](README.md)
