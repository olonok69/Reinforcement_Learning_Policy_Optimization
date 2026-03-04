# Reinforcement Learning: Policy Optimization Methods

This guide summarizes the policy-optimization algorithms implemented in this repository for `CartPole-v1`.

---

## 1. Policy Gradient (REINFORCE)

Policy Gradient directly optimizes a stochastic policy $\pi_\theta(a|s)$ by maximizing expected return:

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

A common gradient estimator is:

$$
\nabla_\theta J(\theta) \propto \mathbb{E}[\nabla_\theta\log\pi_\theta(a_t|s_t)\,\hat{A}_t]
$$

- Main benchmark module: `benchmarks/policy_gradient.py`
- Standalone runner: `policy_gradient_benchmark.py`

---

## 2. A2C (Advantage Actor-Critic)

A2C combines:
- **Actor**: policy network that selects actions
- **Critic**: value estimator used to reduce gradient variance

The advantage term $\hat{A}_t$ stabilizes updates compared with plain REINFORCE.

- Main benchmark module: `benchmarks/a2c.py`
- Standalone runner: `a2c_benchmark.py`

---

## 3. A3C (Asynchronous Advantage Actor-Critic)

A3C extends actor-critic with multiple workers collecting rollouts asynchronously and updating shared parameters.

- Main benchmark module: `benchmarks/a3c.py`
- Standalone runner: `a3c_benchmark.py`

---

## 4. PPO (Proximal Policy Optimization)

PPO improves training stability using a clipped surrogate objective to limit large policy updates.

- Main benchmark module: `benchmarks/ppo.py`
- Standalone runner: `ppo_benchmark.py`

---

## 5. TRPO (Trust Region Policy Optimization)

TRPO performs constrained policy updates inside a trust region for robust improvement steps.

- Main benchmark module: `benchmarks/trpo.py`
- Standalone runner: `trpo_benchmark.py`

---

## Comparison Summary

| Feature | REINFORCE | A2C/A3C | PPO | TRPO |
| :--- | :--- | :--- | :--- | :--- |
| **Stability** | Low-Medium | Medium | High | High |
| **Implementation complexity** | Low | Medium | Medium | High |
| **Compute cost** | Low-Medium | Medium | Medium | High |
| **Typical practical baseline** | Educational | Strong baseline | Strong baseline | Advanced baseline |
