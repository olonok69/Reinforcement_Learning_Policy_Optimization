from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from benchmarks.common import record_policy_video
from benchmarks.trpo_core import backtracking_line_search, conjugate_gradients, flat_grad, flat_params, set_flat_params


@dataclass
class TRPONativeConfig:
    env_name: str = "CartPole-v1"
    total_timesteps: int = 100_000
    batch_size: int = 2_048
    gamma: float = 0.99
    gae_lambda: float = 0.97
    max_kl: float = 1e-2
    damping: float = 1e-1
    cg_steps: int = 10
    value_lr: float = 1e-3
    value_iters: int = 40
    hidden_size: int = 128
    record_video: bool = False
    video_dir: str = "videos/trpo_native"
    video_episodes: int = 3


class CategoricalPolicy(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class ValueNet(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states).squeeze(-1)


def _compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        not_done = 1.0 - float(dones[t])
        next_v = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * not_done - values[t]
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def run_trpo_native(config: TRPONativeConfig | None = None) -> list[float]:
    cfg = config or TRPONativeConfig()
    print("\n--- Starting TRPO (Native) ---")

    env = gym.make(cfg.env_name)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = CategoricalPolicy(obs_size, n_actions, hidden_size=cfg.hidden_size)
    value_net = ValueNet(obs_size, hidden_size=cfg.hidden_size)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=cfg.value_lr)

    rewards_history: list[float] = []
    timestep_count = 0
    state, _ = env.reset()
    episode_reward = 0.0

    while timestep_count < cfg.total_timesteps:
        states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        dones: list[bool] = []
        old_log_probs: list[float] = []
        values: list[float] = []

        collected = 0
        while collected < cfg.batch_size and timestep_count < cfg.total_timesteps:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = policy(state_t)
                dist = torch.distributions.Categorical(logits=logits)
                action_t = dist.sample()
                log_prob_t = dist.log_prob(action_t)
                value_t = value_net(state_t)

            action = int(action_t.item())
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(np.asarray(state, dtype=np.float32))
            actions.append(action)
            rewards.append(float(reward))
            dones.append(done)
            old_log_probs.append(float(log_prob_t.item()))
            values.append(float(value_t.item()))

            episode_reward += float(reward)
            state = next_state
            collected += 1
            timestep_count += 1

            if done:
                rewards_history.append(episode_reward)
                episode_reward = 0.0
                state, _ = env.reset()

        if not states:
            continue

        with torch.no_grad():
            next_value = float(value_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).item())

        states_np = np.asarray(states, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.int64)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.bool_)
        values_np = np.asarray(values, dtype=np.float32)
        old_log_probs_np = np.asarray(old_log_probs, dtype=np.float32)

        adv_np, returns_np = _compute_gae(
            rewards_np,
            dones_np,
            values_np,
            next_value,
            cfg.gamma,
            cfg.gae_lambda,
        )
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

        states_t = torch.tensor(states_np, dtype=torch.float32)
        actions_t = torch.tensor(actions_np, dtype=torch.int64)
        old_log_probs_t = torch.tensor(old_log_probs_np, dtype=torch.float32)
        advantages_t = torch.tensor(adv_np, dtype=torch.float32)
        returns_t = torch.tensor(returns_np, dtype=torch.float32)

        for _ in range(cfg.value_iters):
            pred_values = value_net(states_t)
            value_loss = nn.functional.mse_loss(pred_values, returns_t)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

        with torch.no_grad():
            old_logits = policy(states_t)
            old_dist = torch.distributions.Categorical(logits=old_logits)
            old_probs = old_dist.probs.detach()

        params = tuple(policy.parameters())

        def surrogate_loss() -> torch.Tensor:
            logits = policy(states_t)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            return -(ratio * advantages_t).mean()

        def mean_kl() -> torch.Tensor:
            logits = policy(states_t)
            logp_new = torch.log_softmax(logits, dim=1)
            logp_old = torch.log(old_probs + 1e-8)
            kl = (old_probs * (logp_old - logp_new)).sum(dim=1)
            return kl.mean()

        loss = surrogate_loss()
        grads = torch.autograd.grad(loss, params)
        loss_grad = flat_grad(grads, params).detach()

        def fisher_vector_product(vec: torch.Tensor) -> torch.Tensor:
            kl = mean_kl()
            kl_grads = torch.autograd.grad(kl, params, create_graph=True)
            flat_kl_grad = flat_grad(kl_grads, params)
            kl_v = (flat_kl_grad * vec).sum()
            hvp_grads = torch.autograd.grad(kl_v, params)
            hvp = flat_grad(hvp_grads, params).detach()
            return hvp + cfg.damping * vec

        step_dir = conjugate_gradients(fisher_vector_product, -loss_grad, cfg.cg_steps)
        shs = 0.5 * (step_dir * fisher_vector_product(step_dir)).sum()
        lm = torch.sqrt(torch.clamp(shs / cfg.max_kl, min=1e-8))
        full_step = step_dir / (lm + 1e-8)

        prev = flat_params(policy).detach().clone()

        def evaluate_current() -> tuple[torch.Tensor, torch.Tensor]:
            new_loss = surrogate_loss().detach()
            new_kl = mean_kl().detach()
            return new_loss, new_kl

        success, candidate = backtracking_line_search(
            policy=policy,
            prev_params=prev,
            full_step=full_step,
            evaluate=evaluate_current,
        )

        with torch.no_grad():
            if success:
                set_flat_params(policy, candidate)
            if mean_kl().item() > cfg.max_kl:
                set_flat_params(policy, prev)

    if cfg.record_video:
        policy.eval()

        def _policy(state_np: np.ndarray) -> int:
            with torch.no_grad():
                state_t = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
                logits = policy(state_t)
                return int(torch.argmax(logits, dim=1).item())

        record_policy_video(
            env_name=cfg.env_name,
            video_dir=cfg.video_dir,
            episodes=cfg.video_episodes,
            name_prefix="trpo_native",
            policy_fn=_policy,
        )

    env.close()
    return rewards_history
