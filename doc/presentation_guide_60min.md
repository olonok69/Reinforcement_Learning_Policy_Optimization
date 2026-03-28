# RL Policy Optimization — Presentation Guide Part 1 (60 minutes)

## Scope of Part 1
This first session is intentionally detailed for a low-level audience.

It covers:
- RL foundations needed before policy optimization
- Why Monte Carlo estimation appears in REINFORCE
- The policy gradient theorem intuition
- REINFORCE end-to-end (math + implementation mapping)
- How to run and explain the repository code for REINFORCE

It does **not** cover A2C/A3C/PPO/TRPO in depth. Those are in Part 2.

---

## 1) Session objective
By the end of this Part 1 talk, the audience should be able to explain:
- What a policy is and why we optimize it directly
- What return means in episodic tasks
- Why Monte Carlo returns are unbiased but high variance
- How REINFORCE updates policy parameters
- How theory maps to repository code and runnable scripts

---

## 2) Suggested 60-minute agenda

- **0–8 min**: Problem framing + what this app does
- **8–20 min**: RL fundamentals (MDP, trajectory, return)
- **20–33 min**: Policy optimization and policy gradient intuition
- **33–48 min**: REINFORCE detailed walkthrough (equations + pseudo-code + code links)
- **48–55 min**: Live run (single method + compare mode)
- **55–60 min**: Q&A + transition to Part 2

---

## 3) What this application is

This repository is a benchmarking app for policy-optimization methods on `CartPole-v1`.

### Core capabilities
- Standalone scripts per algorithm
- Unified orchestrator for algorithm comparison
- Standardized benchmark metrics and outputs
- Aggregate reporting and plots

### Entry points
- Unified orchestrator: [run_all_comparison.py](../run_all_comparison.py)
- REINFORCE standalone runner: [policy_gradient_benchmark.py](../policy_gradient_benchmark.py)

### Core REINFORCE module for this session
- [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)

---

## 4) RL fundamentals (beginner-first)

### 4.1 MDP building blocks
An RL task is usually modeled as a Markov Decision Process (MDP):
- state: $s_t$
- action: $a_t$
- reward: $r_t$
- transition dynamics: $P(s_{t+1}|s_t,a_t)$
- discount factor: $\gamma \in [0,1]$

### 4.2 Trajectory and return
One episode (trajectory) is:

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\dots)
$$

Discounted return from time $t$:

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}
$$

Low-level intuition:
- Immediate rewards matter more than distant rewards when $\gamma<1$.
- Return is the training target signal that tells us if sampled behavior was good.

### 4.3 Why stochastic policies?
For policy optimization we model:

$$
\pi_\theta(a|s)
$$

as a probability distribution over actions. This helps:
- exploration during training
- smooth gradients for optimization
- learning action preference, not only a hard decision rule

---

## 5) Policy optimization from first principles

### 5.1 Objective
We maximize expected trajectory return:

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

### 5.2 Log-derivative trick (high-level)
Directly differentiating trajectory probability is hard.
Policy gradient uses:

$$
\nabla_\theta p_\theta(\tau)=p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)
$$

which leads to a gradient estimate using sampled trajectories.

### 5.3 Policy gradient estimator shape
A common form:

$$
\nabla_\theta J(\theta)\propto\mathbb{E}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)\,G_t\right]
$$

Interpretation for beginners:
- $\nabla_\theta\log\pi_\theta(a_t|s_t)$: “direction to increase probability of chosen action”
- $G_t$: “how good that sampled outcome turned out to be”
- Multiplication means better outcomes reinforce corresponding action tendencies.

---

## 6) Monte Carlo in REINFORCE

### 6.1 What Monte Carlo means here
REINFORCE waits until episode end, then computes full returns from sampled rewards.
No bootstrap value target is used in the base version.

### 6.2 Why this is useful
- Unbiased estimate of return for sampled policy
- Very simple implementation

### 6.3 Why this is difficult
- High variance updates
- Learning can be unstable and slower

### 6.4 Small numerical example
Suppose one episode has rewards: $[1,1,1]$, with $\gamma=0.9$.
Then returns are:

$$
G_0=1+0.9+0.9^2=2.71,
\quad G_1=1+0.9=1.9,
\quad G_2=1
$$

The earliest action gets highest weight because it influences more future rewards.

---

## 7) REINFORCE algorithm (step-by-step)

1. Reset environment and sample one episode using current policy.
2. Store per-step log-probabilities and rewards.
3. Compute discounted returns for each timestep.
4. (Optional) normalize returns to stabilize scale.
5. Compute loss:

$$
\mathcal{L}_{policy}=-\sum_t\log\pi_\theta(a_t|s_t)\,G_t
$$

6. Backpropagate and update parameters with Adam.
7. Repeat for many episodes.

---

## 8) Code mapping for Part 1

### REINFORCE implementation
- Config/dataclass: [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)
- Policy network (`PolicyNetwork`): [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)
- Discounted returns helper (`_discounted_returns`): [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)
- Training loop (`run_policy_gradient`): [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)

### CLI runner for demo
- [policy_gradient_benchmark.py](../policy_gradient_benchmark.py)

### Unified comparison entrypoint
- [run_all_comparison.py](../run_all_comparison.py)

---

## 9) Demo commands (Part 1)

### Setup
```bash
uv sync
```

### Run REINFORCE only
```bash
uv run python policy_gradient_benchmark.py
```

### REINFORCE with custom knobs
```bash
uv run python policy_gradient_benchmark.py --episodes 900 --gamma 0.99 --learning-rate 0.001
```

### REINFORCE with evaluation video
```bash
uv run python policy_gradient_benchmark.py --record-video --video-dir videos/policy_gradient --video-episodes 2
```

### Through unified orchestrator (REINFORCE only)
```bash
uv run python run_all_comparison.py --methods policy_gradient --policy-gradient-episodes 900
```

---

## 10) Teaching notes for low-level audience

Use this sequence on slides/whiteboard:

1. Define policy as probabilities over actions.
2. Explain one sampled trajectory as “evidence”.
3. Show discounted return and why early actions affect future outcomes.
4. Show policy-gradient term as “increase probability of actions that led to better return”.
5. Clarify Monte Carlo trade-off: simple + unbiased, but noisy.

Common confusion to address explicitly:
- “Is this supervised learning?” → No labels; rewards are delayed and environment-generated.
- “Why not just choose max action immediately?” → Stochasticity supports exploration and gradient learning.
- “Why normalize returns?” → Helps update magnitudes be more stable.

---

## 11) Bridge to Part 2

Part 2 answers the natural question after REINFORCE:
- How do we reduce variance and improve stability/performance?

That leads to:
- A2C / A3C (critic baseline + parallelism)
- PPO (clipped updates)
- TRPO (trust-region updates)

Part 2 guide:
- [presentation_guide_part2_60min.md](presentation_guide_part2_60min.md)

---

## 12) Suggested external reading (concept reinforcement)

Use these as optional references for deeper conceptual reinforcement:
- https://cameronrwolfe.substack.com/p/policy-gradients-the-foundation-of
- https://www.linkedin.com/pulse/policy-gradient-theorem-continuous-tasks-rl-abram-george/
- https://jonathan-hui.medium.com/rl-policy-gradients-explained-9b13b688b146

These references complement this deck; keep repository code as the source-of-truth for implementation details.
