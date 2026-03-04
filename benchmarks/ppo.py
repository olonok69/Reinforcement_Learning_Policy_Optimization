from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class PPOConfig:
    env_name: str = "CartPole-v1"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    episodes: int = 500
    hidden_size: int = 128
    rollout_steps: int = 1024
    update_epochs: int = 4
    minibatch_size: int = 64
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01


class ActorCritic(nn.Module):
    def __init__(self, input_size: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h)


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


def run_ppo(config: PPOConfig | None = None) -> list[float]:
    cfg = config or PPOConfig()
    print("\n--- Starting PPO ---")

    env = gym.make(cfg.env_name)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = ActorCritic(obs_size, n_actions, hidden_size=cfg.hidden_size)
    optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)

    rewards_history: list[float] = []
    state, _ = env.reset()
    ep_reward = 0.0

    while len(rewards_history) < cfg.episodes:
        states: list[np.ndarray] = []
        actions: list[int] = []
        log_probs: list[float] = []
        rewards: list[float] = []
        dones: list[bool] = []
        values: list[float] = []

        for _ in range(cfg.rollout_steps):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, value = net(state_t)
                probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                action_t = dist.sample()
                log_prob_t = dist.log_prob(action_t)

            action = int(action_t.item())
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(np.asarray(state, dtype=np.float32))
            actions.append(action)
            log_probs.append(float(log_prob_t.item()))
            rewards.append(float(reward))
            dones.append(done)
            values.append(float(value.item()))

            ep_reward += float(reward)
            state = next_state

            if done:
                rewards_history.append(ep_reward)
                ep_reward = 0.0
                state, _ = env.reset()
                if len(rewards_history) % 50 == 0:
                    avg_reward = float(np.mean(rewards_history[-50:]))
                    print(f"Episode {len(rewards_history)}, Average Reward (last 50): {avg_reward:.2f}")
                if len(rewards_history) >= cfg.episodes:
                    break

        if not states:
            continue

        with torch.no_grad():
            next_state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            _, next_value_t = net(next_state_t)
            next_value = float(next_value_t.item())

        states_np = np.asarray(states, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.int64)
        old_log_probs_np = np.asarray(log_probs, dtype=np.float32)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.bool_)
        values_np = np.asarray(values, dtype=np.float32)

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
        adv_t = torch.tensor(adv_np, dtype=torch.float32)
        returns_t = torch.tensor(returns_np, dtype=torch.float32)

        data_size = states_t.shape[0]
        idx = np.arange(data_size)

        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, data_size, cfg.minibatch_size):
                mb_idx = idx[start:start + cfg.minibatch_size]
                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                logits, values_pred = net(mb_states)
                probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                values_pred = values_pred.squeeze(-1)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values_pred, mb_returns)

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    env.close()
    return rewards_history
