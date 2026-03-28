from __future__ import annotations
"""Advantage Actor-Critic (A2C) benchmark on CartPole-v1."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.common import record_policy_video


@dataclass
class A2CConfig:
    """Configuration for A2C training."""

    env_name: str = "CartPole-v1"
    gamma: float = 0.99
    learning_rate: float = 1e-3
    episodes: int = 500
    hidden_size: int = 128
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    record_video: bool = False
    video_dir: str = "videos/a2c"
    video_episodes: int = 3


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic network."""

    def __init__(self, input_size: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value


def run_a2c(config: A2CConfig | None = None) -> list[float]:
    """Train an A2C agent and return per-episode rewards."""

    cfg = config or A2CConfig()
    print("\n--- Starting A2C ---")

    env = gym.make(cfg.env_name)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = ActorCritic(obs_size, n_actions, hidden_size=cfg.hidden_size)
    optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)

    episode_rewards: list[float] = []

    for episode in range(cfg.episodes):
        state, _ = env.reset()
        done = False

        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[float] = []
        entropies: list[torch.Tensor] = []

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = net(state_t)

            probs = torch.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            log_probs.append(dist.log_prob(action))
            values.append(value.squeeze(-1).squeeze(0))
            rewards.append(float(reward))
            entropies.append(dist.entropy())
            state = next_state

        returns: list[float] = []
        running_return = 0.0
        for reward in reversed(rewards):
            running_return = reward + cfg.gamma * running_return
            returns.append(running_return)
        returns.reverse()

        returns_t = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        entropy_t = torch.stack(entropies).mean()

        advantages_t = returns_t - values_t.detach()
        policy_loss = -(log_probs_t * advantages_t).mean()
        value_loss = nn.functional.mse_loss(values_t, returns_t)

        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_t

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
                logits, _ = net(state_t)
                return int(torch.argmax(logits, dim=1).item())

        record_policy_video(
            env_name=cfg.env_name,
            video_dir=cfg.video_dir,
            episodes=cfg.video_episodes,
            name_prefix="a2c",
            policy_fn=_policy,
        )

    env.close()
    return episode_rewards
