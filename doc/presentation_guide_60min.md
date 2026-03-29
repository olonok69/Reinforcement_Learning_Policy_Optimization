# RL Policy Optimization — Presentation Guide Part 1 (60 minutes)

## Scope of Part 1
This first session is intentionally detailed for a low-level audience.

It covers:
- RL foundations needed before policy optimization
- Why Monte Carlo estimation appears in REINFORCE
- The policy gradient theorem intuition and full derivation roadmap
- Value functions, advantage functions, and their role in reducing variance
- Policy gradient variants (reward-to-go, baseline, vanilla PG)
- REINFORCE end-to-end (math + implementation mapping)
- Connection to RLHF and modern LLM finetuning (PPO)
- How to run and explain the repository code for REINFORCE

It does **not** cover A2C/A3C/PPO/TRPO in depth. Those are in Part 2.

**Key reference**: [Cameron R. Wolfe — Policy Gradients: The Foundation of RLHF](https://cameronrwolfe.substack.com/p/policy-gradients-the-foundation-of)

---

## 1) Session objective
By the end of this Part 1 talk, the audience should be able to explain:
- What a policy is and why we optimize it directly
- What return means in episodic tasks
- Why Monte Carlo returns are unbiased but high variance
- How the log-derivative trick makes policy gradients computable
- How REINFORCE updates policy parameters
- What the advantage function is and why it matters for A2C/PPO
- How the four policy gradient variants progressively reduce variance
- How theory maps to repository code and runnable scripts
- Why policy gradients are the foundation of RLHF for LLMs

---

## 2) Suggested 60-minute agenda

- **0–8 min**: Problem framing + what this app does
- **8–18 min**: RL fundamentals (MDP, trajectory, return)
- **18–28 min**: Policy optimization objective + log-derivative trick
- **28–38 min**: Policy gradient theorem + value/advantage functions
- **38–48 min**: REINFORCE detailed walkthrough + policy gradient variants
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
- **state** `sₜ` — what the agent observes
- **action** `aₜ` — what the agent does
- **reward** `rₜ` — immediate feedback from the environment
- **transition dynamics** `P(sₜ₊₁ | sₜ, aₜ)` — how the world changes
- **discount factor** `γ ∈ [0, 1]` — how much we value future vs immediate rewards

### 4.2 Trajectory and return
One episode (trajectory) is:

```
τ = (s₀, a₀, r₀, s₁, a₁, r₁, ...)
```

Discounted return from time `t`:

```
Gₜ = rₜ + γ · rₜ₊₁ + γ² · rₜ₊₂ + ...
   = Σₖ γᵏ · rₜ₊ₖ
```

Low-level intuition:
- Immediate rewards matter more than distant rewards when `γ < 1`.
- Return is the training target signal that tells us if sampled behavior was good.

Two types of return (from Wolfe's article):
- **Finite-horizon undiscounted**: sum of all rewards in a fixed-length episode
- **Infinite-horizon discounted**: sum with `γ` to ensure convergence

### 4.3 Why stochastic policies?
For policy optimization we model:

```
πθ(a | s)  =  P(action = a | state = s)
```

as a probability distribution over actions. This helps:
- exploration during training
- smooth gradients for optimization
- learning action preference, not only a hard decision rule

---

## 5) Value-Based vs Policy-Based

Two fundamentally different approaches to RL:

| Aspect | Value-Based (DQN) | Policy-Based (This Series) |
|--------|-------------------|---------------------------|
| Strategy | Learn Q(s,a), pick argmax | Directly optimize π(a\|s) |
| Policy | Implicit (derived from Q) | Explicit (neural net outputs probabilities) |
| Actions | Discrete only (easily) | Continuous and discrete |
| Foundation | Bellman equation | Policy gradient theorem |
| Used in RLHF | No | Yes — PPO is policy optimization |

**Analogy for beginners**:
- Value-based: rating every dish on a menu, then always picking the highest-rated one
- Policy-based: developing a personal taste — you directly learn which dishes you prefer, without rating every single one

---

## 6) Policy optimization from first principles

### 6.1 Objective
We maximize expected trajectory return:

```
J(θ) = E_τ~πθ [ R(τ) ]
```

In words: the average total reward we get when sampling trajectories from policy `π`.

### 6.2 Gradient ascent
To maximize `J(θ)`, we use gradient ascent:

```
θ  ←  θ + α · ∇θ J(θ)
```

At each step:
1. Compute gradient of the objective w.r.t. current parameters
2. Multiply by learning rate `α`
3. Move parameters uphill (add, not subtract — ascent not descent)

### 6.3 The hard part
Computing `∇J(θ)` is hard because the gradient involves `P(τ|θ)` — the probability of a trajectory — which depends on both the policy AND unknown environment dynamics.

**Solution**: The log-derivative trick (next section).

---

## 7) The log-derivative trick (step by step)

This is the key mathematical insight that makes policy gradients practical.
Present this as a 5-step derivation:

### Step 1: The problem
We need:

```
∇J(θ) = ∇ Σ P(τ|θ) · R(τ)
```

But `P(τ|θ)` includes environment dynamics we don't know!

### Step 2: The calculus identity
From basic calculus, the derivative of log is:

```
∇ log f(x)  =  ∇f(x) / f(x)
```

Rearranging:

```
∇f(x)  =  f(x) · ∇ log f(x)
```

### Step 3: Apply to ∇P(τ|θ)

```
∇P(τ|θ)  =  P(τ|θ) · ∇log P(τ|θ)
```

Now `P(τ|θ)` reappears as a multiplier → the sum becomes an **expectation** we can sample!

### Step 4: Environment cancels out
Trajectory probability is a product:

```
P(τ|θ) = P(s₀) · Π πθ(aₜ|sₜ) · P(sₜ₊₁|sₜ,aₜ)
```

Taking log turns products into sums:

```
log P(τ|θ) = log P(s₀) + Σ log πθ(aₜ|sₜ) + Σ log P(sₜ₊₁|sₜ,aₜ)
```

**When we take ∇ w.r.t. θ, only `πθ(a|s)` depends on θ!**
Environment terms `P(s₀)` and `P(s'|s,a)` vanish because their gradient w.r.t. θ is zero.

### Step 5: The result

```
∇J(θ) = E[ Σₜ ∇log πθ(aₜ|sₜ) · Gₜ ]
```

We only need two things we CAN compute:
1. Gradient of log-policy (from neural network — autograd handles this)
2. Return `G` (from sampled episodes — just sum rewards)

**No environment model needed! This is model-free RL.**

---

## 8) Policy gradient theorem — the core equation

### The estimator (practical version)

```
∇J(θ)  ≈  (1/N) Σᵢ [ Σₜ ∇log πθ(aₜ|sₜ) · Gₜ ]
```

We sample `N` trajectories, compute the expression for each, and average.

### Breaking it down — word by word

| Term | Plain English |
|------|--------------|
| `∇log π(a\|s)` | "Direction to increase probability of chosen action `a` in state `s`" |
| `Gₜ` (Return) | "How good was that sampled outcome?" |
| Multiply them | "Good outcomes REINFORCE their actions, bad outcomes diminish them" |
| `(1/N) Σ` | "Average over N sampled trajectories to estimate the expectation" |

### Intuition
- If an action led to **high return** → increase its probability
- If an action led to **low return** → decrease its probability
- The gradient points in the direction that makes good trajectories more likely

This is why the algorithm is called **REINFORCE** — it reinforces actions proportionally to how well they worked.

### In PyTorch code

```python
loss = -(log_probs * returns).sum()
loss.backward()       # autograd computes ∇ for us
optimizer.step()      # gradient ascent (minus sign flips descent to ascent)
```

---

## 9) Value functions and the advantage function

### Four value functions (from Wolfe's article)

| Function | Name | Meaning |
|----------|------|---------|
| `Vπ(s)` | State value | Expected return starting from state `s`, following policy `π`. "How good is it to be here?" |
| `Qπ(s,a)` | Action value | Expected return starting from `s`, taking action `a`, then following `π`. "How good is this specific action here?" |
| `V*(s)` | Optimal state value | Same but assuming the best possible policy |
| `Q*(s,a)` | Optimal action value | Same with optimal policy. **This is what DQN learns.** |

### The advantage function

```
A(s,a)  =  Q(s,a) - V(s)
```

"How much **better** is action `a` compared to the **average** action in state `s`?"

- `A > 0` → better than average → reinforce this action
- `A < 0` → worse than average → discourage this action

### Why advantage matters
REINFORCE uses raw return `G` as weight — noisy because **all** actions get credit for the total episode outcome.
Using advantage instead: only actions that did **BETTER than expected** get positively reinforced.
This dramatically reduces gradient variance.

### The vanilla policy gradient (advantage-based)

```
∇J(θ) = E[ ∇log π(a|s) · A(s,a) ]
```

Same structure as REINFORCE, but `G` is replaced with `A(s,a)`.
**This is what A2C and PPO actually optimize.**

### Connection to RLHF
PPO — the RL algorithm behind ChatGPT — uses exactly this advantage-based policy gradient with clipped updates and GAE (Generalized Advantage Estimation, Schulman 2015).

**Evolution**: `REINFORCE → + baseline V(s) → advantage → A2C → PPO`

### Code mapping

```python
# benchmarks/a2c.py
advantage = returns - value_net(states)
policy_loss = -(log_probs * advantage)
```

---

## 10) Policy gradient variants — reducing variance step by step

Present these four variants as a progression. Each keeps the same expected value but reduces noise:

### Variant 1: Basic (REINFORCE)

```
∇J = E[ Σₜ ∇log π(a|s) · R(τ) ]
```

Weight = **total trajectory return** `R(τ)`.
Problem: past rewards (before action was taken) add noise without useful signal.

### Variant 2: Reward-to-Go

```
∇J = E[ Σₜ ∇log π(aₜ|sₜ) · Gₜ ]
```

Weight = **return from timestep t onward only**.
Past rewards drop out — an action should only be judged by what happens **after** it.
Same expected value, **lower variance**.

> **This is what our `benchmarks/policy_gradient.py` implements via `_discounted_returns()`.**

### Variant 3: Baseline subtraction

```
∇J = E[ Σₜ ∇log π · (Gₜ - b(s)) ]
```

Subtract a baseline `b(s)` that only depends on state.
Common choice: `b(s) = V(s)` (the state-value function).
Only **above-average** actions get positively reinforced.

### Variant 4: Vanilla Policy Gradient (Advantage)

```
∇J = E[ Σₜ ∇log π · Aπ(s,a) ]
```

Use advantage `A = Q(s,a) - V(s)` as weight. **Lowest variance** of all four.

> **This is what A2C (`benchmarks/a2c.py`) and PPO (`benchmarks/ppo.py`) implement.**

### Key mathematical insight
All four have the **SAME expected value** (they are unbiased estimators of the true policy gradient).
The difference is **variance**: each step removes noise without changing what we're optimizing.

```
Lower variance → fewer episodes needed → faster convergence → more stable training
```

### GAE (Generalized Advantage Estimation)
In practice, estimating `A(s,a)` exactly is hard. GAE (Schulman 2015) blends multi-step TD estimates with a `λ` parameter to balance bias and variance. PPO uses GAE by default.

> **Code**: `benchmarks/ppo.py` uses GAE in its rollout buffer computation.

---

## 11) Monte Carlo in REINFORCE

### What Monte Carlo means here
REINFORCE waits until episode end, then computes full returns from sampled rewards.
No bootstrap value target is used in the base version.

### Why this is useful
- Unbiased estimate of return for sampled policy
- Very simple implementation

### Why this is difficult
- High variance updates
- Learning can be unstable and slower

### Small numerical example

Episode rewards: `[1, 1, 1]`, with `γ = 0.9`

```
G₀ = 1 + 0.9×1 + 0.9²×1  = 1 + 0.9 + 0.81  = 2.71
G₁ = 1 + 0.9×1            = 1 + 0.9          = 1.90
G₂ = 1                                        = 1.00
```

The earliest action gets highest weight because it influences more future rewards.

### Return normalization
Common practice: normalize returns per episode.

```python
G_norm = (G - mean(G)) / std(G)
```

This ensures roughly half the actions are encouraged and half are discouraged — stabilizing gradient magnitudes. This acts as a simple baseline approximation.

### Reward-to-go trick
Only future rewards matter for each action:
Use `G` from timestep `t` onward (not from episode start). Reduces variance without adding bias.

---

## 12) REINFORCE algorithm (step-by-step)

```
1. Reset environment and sample one episode using current policy
2. At each step, store log π(aₜ|sₜ) and reward rₜ
3. After episode ends, compute discounted returns:
       Gₜ = rₜ + γ · Gₜ₊₁     (walk backwards)
4. (Optional) normalize returns:
       G = (G - mean) / std
5. Compute policy loss:
       L = -Σₜ log π(aₜ|sₜ) · Gₜ
6. Backpropagate and update parameters with Adam:
       loss.backward()
       optimizer.step()
7. Repeat for many episodes
```

**Why the minus sign?** PyTorch does gradient **descent** by default. We want gradient **ascent** (maximize return). The minus sign flips descent into ascent.

---

## 13) Code mapping for Part 1

### REINFORCE implementation
- Config/dataclass: [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)
- Policy network (`PolicyNetwork`): 2-layer MLP (128 hidden units, ReLU). Input: state vector. Output: action logits → softmax → Categorical distribution.
- Discounted returns helper (`_discounted_returns`): Walks rewards backward computing `G = r + γ*G_next`. Implements the **reward-to-go** variant.
- Training loop (`run_policy_gradient`): Each episode: sample trajectory → compute returns → normalize → compute loss → backprop → step optimizer.
- `PolicyGradientConfig`: `gamma=0.99, lr=1e-3, episodes=700, hidden_size=128, normalize_returns=True`

### Theory-to-code correspondence

| Theory concept | Code location |
|---------------|--------------|
| `πθ(a\|s)` — stochastic policy | `PolicyNetwork.forward()` → `Categorical(logits)` |
| `log π(a\|s)` — log probability | `dist.log_prob(action)` |
| `Gₜ` — discounted return (reward-to-go) | `_discounted_returns()` |
| Normalization (baseline approx.) | `(returns - mean) / std` in training loop |
| `L = -Σ log π · G` — policy loss | `-(log_probs * returns).sum()` |
| `∇θ` via autograd | `loss.backward(); optimizer.step()` |

---

## 14) Demo commands (Part 1)

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

## 15) Teaching notes for low-level audience

Use this sequence on slides/whiteboard:

1. Define policy as probabilities over actions.
2. Explain one sampled trajectory as "evidence".
3. Show discounted return and why early actions affect future outcomes.
4. Show policy-gradient term as "increase probability of actions that led to better return".
5. Introduce value functions `V(s)` and `Q(s,a)` as "what we expect on average".
6. Show advantage `A = Q - V` as "was this action better or worse than average?"
7. Walk through the four variants: REINFORCE → reward-to-go → baseline → advantage.
8. Clarify Monte Carlo trade-off: simple + unbiased, but noisy.
9. Connect to RLHF: "PPO uses the advantage-based variant + clipping".

**Common confusion to address explicitly:**
- "Is this supervised learning?" → No labels; rewards are delayed and environment-generated.
- "Why not just choose max action immediately?" → Stochasticity supports exploration and gradient learning.
- "Why normalize returns?" → Helps update magnitudes be more stable. Acts as a simple baseline.
- "What's the difference between G and A?" → G is raw return, A is return minus baseline (how much better than expected).

---

## 16) Bridge to Part 2

Part 2 answers the natural question after REINFORCE:
- How do we reduce variance and improve stability/performance?

REINFORCE weaknesses → Part 2 solutions:

| # | REINFORCE Weakness | Part 2 Solution |
|---|-------------------|----------------|
| 1 | High variance gradients | **A2C**: add critic V(s) to compute advantage |
| 2 | Sample inefficiency | **A3C**: parallel workers for data decorrelation |
| 3 | No credit assignment | **PPO**: clipped surrogate + GAE |
| 4 | Sensitive to hyperparameters | **TRPO**: KL-constrained trust region |
| 5 | No update size constraint | **PPO**: clips ratio to prevent catastrophic updates |

All Part 2 methods build directly on the policy gradient foundation covered today.