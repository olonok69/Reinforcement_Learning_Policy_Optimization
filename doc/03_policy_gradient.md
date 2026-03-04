# Policy Gradient (REINFORCE)

## Core idea
Instead of learning $Q(s,a)$, policy gradients directly optimize a parameterized stochastic policy $\pi_\theta(a|s)$ by maximizing expected return:

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

REINFORCE update (Monte Carlo):

$$
\nabla_\theta J(\theta) \approx \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\,G_t
$$

Interpretation:
- $\log \pi_\theta(a_t|s_t)$ increases probability of sampled actions.
- $G_t$ acts as a quality signal for those actions.
- Good trajectories amplify their own action probabilities; bad trajectories suppress them.

Because $G_t$ is estimated from full-episode returns, variance is often high, especially early in training.

## How this repo implements it
- Module: `benchmarks/policy_gradient.py`
- Runner: `policy_gradient_benchmark.py`
- Environment: `CartPole-v1`
- Uses episode returns, optional return normalization.

## Training flow
1. Roll out one full episode with stochastic actions.
2. Compute discounted returns per timestep.
3. Weight log-probabilities by returns.
4. Backpropagate policy loss.

In practice, this is an unbiased but noisy estimator of the policy gradient.

## Why it works (in one paragraph)
The log-derivative trick converts gradients over trajectory probabilities into an expectation of score-function terms, making it possible to optimize expected return with sampled rollouts. REINFORCE is the most direct expression of this idea, which is why it is often used as the conceptual starting point for actor-critic and PPO-style methods.

## Strengths
- Handles stochastic policies naturally.
- Works well in continuous and high-dimensional policy spaces.
- Simple conceptual bridge to actor-critic methods.

## Limitations
- High-variance gradients.
- Sample inefficient (full-episode Monte Carlo).
- Sensitive to baseline/normalization choices.

## Common tuning levers
- **Learning rate**: too high causes unstable oscillation; too low slows learning.
- **Discount factor $\gamma$**: higher values emphasize long-horizon effects but increase variance.
- **Return normalization**: improves optimization scale consistency.
- **Entropy coefficient** (if enabled): raises exploration, but too large slows convergence.

## Practical notes
- Return normalization usually improves stability.
- Entropy regularization (not mandatory) can improve exploration.
- A2C/PPO are usually more stable successors in practice.
