# RL Algorithm Explanations (Code-Referenced)

This document explains, in a practical way, how the demos in this repository work and how they map to core Reinforcement Learning theory:

- Tabular Q-learning on FrozenLake
- REINFORCE (policy gradient) on CartPole
- PPO on CartPole with Stable-Baselines3

The goal is to help you move from theory to implementation.

**Key reference**: [Cameron R. Wolfe — Policy Gradients: The Foundation of RLHF](https://cameronrwolfe.substack.com/p/policy-gradients-the-foundation-of)

---

## 0) 60-minute teaching plan (approx.)

- **0-8 min**: RL fundamentals (agent, environment, state, action, reward, policy).
- **8-15 min**: Model-free vs model-based RL.
- **15-30 min**: Q-learning (idea + code mapping).
- **30-45 min**: REINFORCE (idea + code mapping).
- **45-55 min**: PPO in practice with Stable-Baselines3.
- **55-60 min**: Wrap-up: model vs policy vs value function.

---

## 1) RL foundations

Core components:

- **Agent**: the learner and decision-maker.
- **Environment**: the system the agent interacts with.
- **State** `s`: the current situation.
- **Action** `a`: a possible decision.
- **Reward** `r`: a feedback signal.
- **Policy** `π`: a rule for choosing actions.

General objective:

- Maximize expected cumulative discounted return.

---

## 2) Model-free vs model-based RL

- **Model-based RL**: uses/learns transition and reward dynamics for planning.
- **Model-free RL**: learns directly from experience without an explicit environment model.

This repository focuses on **model-free** methods for teaching clarity:

- Q-learning (value-based, tabular in this project).
- REINFORCE and PPO (policy optimization methods).

---

## 3) Q-learning

Main file: `demos/q_learning_frozenlake.py`

### 3.1 Intuition

Q-learning learns `Q(s,a)`, which estimates how good each action is in each state. The policy is derived with `argmax`.

### 3.2 Update rule

```
Q(s,a)  ←  Q(s,a) + α · [ r + γ · max_a' Q(s',a') - Q(s,a) ]
```

Where:

- `α`: learning rate
- `γ`: discount factor
- `r + γ · max Q(s',a')`: the Bellman target (what Q *should* be)
- `r + γ · max Q(s',a') - Q(s,a)`: the TD error (how wrong we were)

### 3.3 What the script does

- Initializes the **Q-table** (all zeros).
- Selects actions using **epsilon-greedy** exploration.
- Updates values with TD/Bellman targets.
- Decays `epsilon` over time (less exploration, more exploitation).
- Evaluates the final greedy policy and reports success rate.

### 3.4 Code blocks to locate

- Action selection: `epsilon_greedy_action(...)`
- Training loop: `train(...)`
- Final evaluation: `evaluate_policy(...)`

### 3.5 Exploration in Q-learning: epsilon-greedy

Q-learning uses an **explicit, external** exploration strategy:

```python
if random.random() < epsilon:
    action = env.action_space.sample()    # explore: random action
else:
    action = np.argmax(Q[state])          # exploit: best known action
```

Epsilon follows a **manual schedule**:
- Start: `ε = 1.0` → purely random
- Decay: `ε *= decay_rate` each episode
- End: `ε = 0.01` → almost purely greedy

The exploration mechanism is **completely separate** from the Q-values. Even if Q says "left is clearly better", the epsilon coin flip can still force a random action. This ensures every action keeps getting tried.

---

## 4) REINFORCE (policy gradient)

Main file: `demos/reinforce_cartpole.py`

### 4.1 Intuition

REINFORCE learns a stochastic policy `πθ(a|s)` directly (not a Q-table). The policy is a neural network that outputs **probabilities** over actions.

### 4.2 Objective (common form)

```
L(θ) = -Σₜ log πθ(aₜ|sₜ) · Gₜ
```

where `Gₜ` is the discounted return from step `t`.

### 4.3 What the script does

- Defines a policy network (`PolicyNetwork`) with `softmax` output.
- Samples actions from `Categorical(probs)`.
- Stores log-probabilities and rewards for each episode.
- Computes returns with `discounted_returns(...)`.
- Normalizes returns for more stable training.
- Optimizes with backpropagation and `optimizer.step()`.

### 4.4 Code blocks to locate

- Policy model: `PolicyNetwork`
- Discounted returns: `discounted_returns(...)`
- Training loop: `train(...)`

### 4.5 How the neural network actually learns the policy

The process has **three phases** repeated every episode:

**Phase 1 — The network generates probabilities**

When the network receives a state (e.g. in CartPole: cart position, velocity, pole angle, angular velocity), it produces **logits** — a raw number per action:

```python
logits = net(state_t)          # e.g. tensor([0.3, -0.8])
probs = torch.softmax(logits)  # e.g. tensor([0.75, 0.25])
```

The internal weights of `nn.Linear` layers determine which logits come out. At the start, weights are **random**, so the network produces nearly equal probabilities — like an agent that knows nothing.

**Phase 2 — The agent plays a full episode**

With those probabilities, the agent **samples** actions (it doesn't always pick the most probable — this is key for exploration):

```python
dist = Categorical(probs)
action = dist.sample()                        # sample according to probs
log_probs.append(dist.log_prob(action))       # save log π(a|s) for later
```

**Phase 3 — The gradient adjusts the weights**

After the episode ends, returns are computed and the loss is built:

```python
returns = _discounted_returns(rewards, gamma)
returns_t = (returns_t - mean) / std           # normalize

loss = -(log_probs * returns_t).sum()          # policy loss
loss.backward()                                # autograd computes ∇
optimizer.step()                               # adjust weights
```

The product `log_prob × return` tells each network weight: "the action you chose in that state led to a return of X". The gradient pushes the weights in the direction that would make that action **more probable** if the return was high, or **less probable** if the return was low.

**The complete cycle across training:**

```
Episode 1:    random weights → near-random actions    → reward ~20
Episode 50:   slightly tuned → somewhat better        → reward ~80
Episode 300:  well calibrated → mostly correct actions → reward ~400
Episode 500:  converged → near-optimal policy          → reward 500 (max)
```

### 4.6 Can we do this WITHOUT a neural network?

**Yes, absolutely.** The neural network is just one way to represent `π(a|s)`. Alternatives:

**Option 1: Direct table (softmax tabular)**

If the state space is **discrete and small**, you can use a parameter table:

```python
# For an environment with 16 states and 4 actions:
theta = np.zeros((16, 4))       # parameter table

def policy(state):
    logits = theta[state]                              # row from table
    probs = np.exp(logits) / np.exp(logits).sum()      # softmax
    return np.random.choice(4, p=probs)
```

The update is identical in spirit: `θ[s,a] += α · Gₜ · (1 - π(a|s))`. REINFORCE works exactly the same — you just adjust table entries instead of `nn.Linear` weights.

**Option 2: Linear function (no hidden layers)**

```python
W = np.random.randn(n_actions, obs_dim) * 0.01
b = np.zeros(n_actions)

def policy(state):
    logits = W @ state + b
    probs = softmax(logits)
    return np.random.choice(n_actions, p=probs)
```

Equivalent to a single-layer network without activation. Works for simple problems but can't capture non-linear patterns.

**Why use a neural network then?**

| Representation | Advantage | Limitation |
|---------------|-----------|-----------|
| Table | Exact, simple | Only discrete and few states. CartPole has continuous states → impossible |
| Linear | Fast, few parameters | Cannot capture non-linear state-action relationships |
| Neural network | Approximates any function, handles continuous states | More parameters, slower, needs hyperparameter tuning |

CartPole has **continuous states** (angle, velocity are real numbers, not categories), so a direct table doesn't work — you can't have a row for every possible value of 0.0347° angle. The neural network **generalizes**: if it learned "negative angle → push left", it applies that to negative angles it has never seen before.

**The key point:** The **algorithm** (REINFORCE, A2C, PPO) is independent of **how you represent the policy**. What you need is: (1) a differentiable function `π(a|s)`, and (2) the ability to compute `log π(a|s)` and its gradient. Neural networks, tables, and linear functions all satisfy this.

### 4.7 Exploration-exploitation in policy gradients

This is one of the most important differences from Q-learning. In REINFORCE, exploration is **intrinsic** — built into the policy itself, not added externally.

**How it works:**

```python
probs = torch.softmax(logits, dim=1)       # network outputs probabilities
dist = torch.distributions.Categorical(probs)
action = dist.sample()                      # ← exploration happens HERE
```

If the network produces `probs = [0.7, 0.3]`, `dist.sample()` picks "left" 70% of the time and "right" 30%. That 30% **is** the exploration — the agent tries the less-favored action naturally.

**How exploration evolves during training:**

```
Episode 1:    probs ≈ [0.52, 0.48]   → nearly random (lots of exploration)
Episode 100:  probs ≈ [0.70, 0.30]   → prefers one action but still tries both
Episode 300:  probs ≈ [0.85, 0.15]   → fairly confident but not 100%
Episode 600:  probs ≈ [0.95, 0.05]   → almost always exploits the best action
```

The network **self-regulates**: early on, random weights produce similar logits → similar probabilities → lots of exploration. As the gradient reinforces good actions, probabilities concentrate → more exploitation. **No epsilon schedule needed.**

**Risk: premature deterministic collapse**

If the network becomes too confident too early (e.g. `[0.99, 0.01]` before exploring enough), it stops trying the other action and may get stuck in a suboptimal solution.

Basic REINFORCE has **no protection** against this. But A2C and PPO add an **entropy bonus**:

```python
# benchmarks/a2c.py and benchmarks/ppo.py
entropy = dist.entropy().mean()
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
#                                              ^^^^^^^^^^^^^^^^^^^^^^^^
#                              this PENALIZES overly concentrated probabilities
```

Entropy measures how "spread out" a distribution is:

```
probs = [0.50, 0.50]  →  entropy = 0.69  (maximum — fully random)
probs = [0.80, 0.20]  →  entropy = 0.50  (somewhat concentrated)
probs = [0.99, 0.01]  →  entropy = 0.06  (nearly deterministic)
```

By subtracting `entropy_coef × entropy` from the loss (the minus sign turns it into a reward), the optimizer **resists** probability collapse. It's like telling the network "I reward you a little for keeping some uncertainty".

---

## 5) PPO in practice

Main file: `demos/gymnasium_ppo_cartpole.py`

### 5.1 Intuition

PPO (Proximal Policy Optimization) is a modern policy optimization method that is typically more stable than vanilla policy gradients. It builds directly on the advantage-based policy gradient by adding **clipped updates** that prevent destructive large policy changes.

### 5.2 Core idea — clipped surrogate objective

```
L_clip(θ) = E[ min( rₜ(θ) · Aₜ ,  clip(rₜ(θ), 1-ε, 1+ε) · Aₜ ) ]

where  rₜ(θ) = πθ(aₜ|sₜ) / πθ_old(aₜ|sₜ)    (probability ratio)
```

If the new policy changes action probability too much, clipping truncates the incentive to keep pushing. This creates a soft "trust region" without solving constrained optimization.

### 5.3 What the script does

- Creates `CartPole-v1` with monitoring.
- Instantiates `PPO` (`MlpPolicy`) with CLI hyperparameters.
- Trains with `model.learn(total_timesteps=...)`.
- Saves the trained model under `models/`.
- Evaluates deterministic policy performance and reports metrics.

### 5.4 Visual verification and recording

- Human render: `--render-eval --render-episodes N`
- Evaluation video: `--record-video --video-episodes N --video-dir <path>`
- Combined mode: `--record-and-render`

### 5.5 PPO exploration

PPO uses the same **intrinsic stochastic exploration** as REINFORCE (sampling from `Categorical(probs)`) plus an **entropy bonus** (`entropy_coef = 0.01` by default) that actively prevents the policy from becoming too deterministic too early. Additionally, PPO estimates advantages using **GAE** (Generalized Advantage Estimation, λ=0.95), which blends multi-step TD estimates to balance bias and variance — giving cleaner signal than raw Monte Carlo returns.

---

## 6) Value functions and the advantage function

### Four value functions

| Function | Name | Meaning |
|----------|------|---------|
| `Vπ(s)` | State value | Expected return from state `s`, following policy `π`. "How good is it to be here?" |
| `Qπ(s,a)` | Action value | Expected return from `s`, taking action `a`, then following `π`. "How good is this specific action?" |
| `V*(s)` | Optimal state value | Same but assuming the best possible policy |
| `Q*(s,a)` | Optimal action value | Same with optimal policy. **This is what DQN learns.** |

### The advantage function

```
A(s,a) = Q(s,a) - V(s)
```

"How much **better** is action `a` compared to the **average** action in state `s`?"

- `A > 0` → better than average → reinforce this action
- `A < 0` → worse than average → discourage this action

### Why advantage matters

REINFORCE uses raw return `G` as weight — noisy because **all** actions get credit for the total episode outcome. Using advantage instead: only actions that performed **better than expected** get positively reinforced. This dramatically reduces gradient variance.

### Code mapping

```python
# benchmarks/a2c.py — the advantage computation
advantage = returns - value_net(states)          # A = G - V(s)
policy_loss = -(log_probs * advantage).sum()     # advantage-weighted gradient
```

---

## 7) Policy gradient variants — reducing variance step by step

All four variants have the **same expected value** (they are unbiased estimators). The difference is **variance** — each step removes noise without changing what we optimize:

### Variant 1: Basic (REINFORCE)

```
∇J = E[ Σₜ ∇log π(a|s) · R(τ) ]
```

Weight = total trajectory return `R(τ)`. Problem: past rewards add noise.

### Variant 2: Reward-to-Go

```
∇J = E[ Σₜ ∇log π(aₜ|sₜ) · Gₜ ]
```

Weight = return from timestep `t` onward only. Past rewards drop out.
**Implemented in**: `_discounted_returns()` in `benchmarks/policy_gradient.py`.

### Variant 3: Baseline subtraction

```
∇J = E[ Σₜ ∇log π · (Gₜ - b(s)) ]
```

Subtract a baseline `b(s) = V(s)`. Only above-average actions get reinforced.
**Approximate version**: return normalization `(G - mean) / std` in REINFORCE.

### Variant 4: Vanilla Policy Gradient (Advantage)

```
∇J = E[ Σₜ ∇log π · Aπ(s,a) ]
```

Use advantage `A = Q(s,a) - V(s)`. Lowest variance of all four.
**Implemented in**: `benchmarks/a2c.py` and `benchmarks/ppo.py`.

```
Lower variance → fewer episodes needed → faster convergence → more stable training
```

### Connection to RLHF

PPO — the RL algorithm behind ChatGPT — uses the advantage-based variant (Variant 4) with clipped updates and GAE. The entire RLHF pipeline is built on this policy gradient foundation.

**Evolution**: `REINFORCE → reward-to-go → + baseline V(s) → advantage → A2C → PPO`

---

## 8) Q-learning vs REINFORCE (detailed comparison)

### What each learns

- **Q-learning**: state-action values `Q(s,a)`. Policy is implicit: `π(s) = argmax_a Q(s,a)`.
- **REINFORCE**: policy directly `πθ(a|s)`. No value table needed.

### Update style

- **Q-learning**: temporal-difference bootstrapping (1-step target).
- **REINFORCE**: Monte Carlo policy gradient over full episodes.

### Exploration-exploitation comparison

| Aspect | Q-Learning / DQN | REINFORCE | A2C / PPO |
|--------|------------------|-----------|-----------|
| Mechanism | Epsilon-greedy (external) | Sampling from distribution (intrinsic) | Sampling + entropy bonus |
| Controlled by | `ε` (manual schedule) | Network weights (automatic) | Weights + `entropy_coef` |
| At start | `ε=1.0` → random | Random weights → probs ≈ uniform | Same + high entropy rewarded |
| At end | `ε=0.01` → near greedy | Concentrated probs → near greedy | Concentrated but not collapsed |
| Risk | ε decays too fast → underexplores | Premature deterministic collapse | Low (entropy prevents it) |
| Schedule needed | Yes (ε decay) | No | No |

### Typical fit

- **Tabular Q-learning**: small/discrete state spaces (FrozenLake).
- **REINFORCE/PPO with neural networks**: continuous or high-dimensional observations (CartPole, Atari, LLM finetuning).

---

## 9) Key hyperparameters

### Q-learning

- `alpha` — learning rate for Q-value updates
- `gamma` — discount factor
- `epsilon-start`, `epsilon-min`, `epsilon-decay` — exploration schedule
- `episodes`, `max-steps`

### REINFORCE

- `lr` — learning rate for network weight updates
- `gamma` — discount factor for returns
- `episodes` — number of training episodes
- `solve-score` — target reward to consider solved

### PPO

- `timesteps` — total environment interactions
- `learning-rate` — optimizer step size
- `n-steps` — rollout length before each update
- `batch-size` — minibatch size for updates
- `gamma` — discount factor
- `clip-eps` — clipping range for surrogate objective (default 0.2)
- `entropy-coef` — entropy bonus weight (exploration pressure)
- `eval-episodes` — episodes for post-training evaluation

---

## 10) Post-training verification (all 3 demos)

All demos support:

1. **Human render** for visual behavior checks.
2. **MP4 recording** for documentation or teaching material.
3. **Render + recording** in a single run.

Default video directories:

- `videos/q_learning_frozenlake`
- `videos/reinforce_cartpole`
- `videos/ppo_cartpole`

Dependency note:

- If video support is missing, install `moviepy` with:

```powershell
uv pip install moviepy
```

---

## 11) Model vs policy vs value function

- **Model**: "what happens next?" (dynamics/reward prediction).
- **Policy**: "what should I do now?" (action selection).
- **Value function**: "how good is this state/action?" (expected return).

In this repository:

- Q-learning mainly learns a **value function**.
- REINFORCE and PPO mainly optimize a **policy**.
- PPO also uses value estimation internally (critic head for advantage).

---

## 12) Quick commands

### Q-learning

```powershell
python demos/q_learning_frozenlake.py
python demos/q_learning_frozenlake.py --slippery
python demos/q_learning_frozenlake.py --render-eval --render-episodes 5
python demos/q_learning_frozenlake.py --record-video --video-episodes 5 --video-dir videos/q_learning_frozenlake
```

### REINFORCE

```powershell
python demos/reinforce_cartpole.py
python demos/reinforce_cartpole.py --episodes 1500 --lr 0.0005 --gamma 0.99
python demos/reinforce_cartpole.py --render-eval --render-episodes 3
python demos/reinforce_cartpole.py --record-video --video-episodes 3 --video-dir videos/reinforce_cartpole
```

### PPO

```powershell
python demos/gymnasium_ppo_cartpole.py
python demos/gymnasium_ppo_cartpole.py --render-eval --render-episodes 3
python demos/gymnasium_ppo_cartpole.py --record-video --video-episodes 3 --video-dir videos/ppo_cartpole
```