from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class PolicyGradientConfig:
    env_name: str = "CartPole-v1"
    gamma: float = 0.99
    learning_rate: float = 1e-3
    episodes: int = 700
    hidden_size: int = 128
    normalize_returns: bool = True


class PolicyNetwork(nn.Module):
    def __init__(self, input_size: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    out: list[float] = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        out.append(running)
    out.reverse()
    return out


def run_policy_gradient(config: PolicyGradientConfig | None = None) -> list[float]:
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

    env.close()
    return episode_rewards
