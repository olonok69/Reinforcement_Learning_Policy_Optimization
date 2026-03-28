from __future__ import annotations
"""Policy Gradient (REINFORCE) benchmark on CartPole-v1.

This module provides a compact, reproducible baseline for direct policy
optimization using Monte Carlo returns.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.common import record_policy_video


@dataclass
class PolicyGradientConfig:
    """Configuration for REINFORCE training."""

    env_name: str = "CartPole-v1"
    gamma: float = 0.99
    learning_rate: float = 1e-3
    episodes: int = 700
    hidden_size: int = 128
    normalize_returns: bool = True
    record_video: bool = False
    video_dir: str = "videos/policy_gradient"
    video_episodes: int = 3


class PolicyNetwork(nn.Module):
    """Small MLP policy that outputs action logits."""

    def __init__(self, input_size: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Compute discounted returns for one episode."""

    out: list[float] = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        out.append(running)
    out.reverse()
    return out


def run_policy_gradient(config: PolicyGradientConfig | None = None) -> list[float]:
    """Train a REINFORCE agent and return per-episode rewards."""

    cfg = config or PolicyGradientConfig()
    print("\n--- Starting Policy Gradient (REINFORCE) ---")

    env = gym.make(cfg.env_name)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = PolicyNetwork(obs_size, n_actions, hidden_size=cfg.hidden_size)
    optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)

    episode_rewards: list[float] = []

    for episode in range(cfg.episodes):
        state, _ = env.reset()
        done = False

        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = net(state_t)
            probs = torch.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            log_probs.append(dist.log_prob(action))
            rewards.append(float(reward))
            state = next_state

        returns = _discounted_returns(rewards, cfg.gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        if cfg.normalize_returns and len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        policy_loss = -torch.stack(log_probs) * returns_t
        loss = policy_loss.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ep_reward = float(np.sum(rewards))
        episode_rewards.append(ep_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = float(np.mean(episode_rewards[-50:]))
            print(f"Episode {episode + 1}, Average Reward (last 50): {avg_reward:.2f}")

    if cfg.record_video:
        net.eval()

        def _policy(state: np.ndarray) -> int:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits = net(state_t)
                return int(torch.argmax(logits, dim=1).item())

        record_policy_video(
            env_name=cfg.env_name,
            video_dir=cfg.video_dir,
            episodes=cfg.video_episodes,
            name_prefix="policy_gradient",
            policy_fn=_policy,
        )

    env.close()
    return episode_rewards
