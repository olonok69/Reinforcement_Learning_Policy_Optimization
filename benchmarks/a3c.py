from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import queue
import traceback
import typing as tt

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class A3CConfig:
    env_name: str = "CartPole-v1"
    gamma: float = 0.99
    learning_rate: float = 1e-3
    episodes: int = 500
    hidden_size: int = 128
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    workers: int = 4
    rollout_steps: int = 5


class ActorCritic(nn.Module):
    def __init__(self, input_size: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.shared(x)
        return self.policy_head(feats), self.value_head(feats)


def _snapshot_state_dict(shared_state: tt.Any) -> dict[str, torch.Tensor]:
    return {k: v for k, v in shared_state.items()}


def _worker_loop(
    worker_id: int,
    cfg: A3CConfig,
    state_dict: dict,
    data_queue: tt.Any,
    error_queue: tt.Any,
    stop_event: tt.Any,
) -> None:
    env = None
    try:
        env = gym.make(cfg.env_name)
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n

        net = ActorCritic(obs_size, n_actions, hidden_size=cfg.hidden_size)
        net.load_state_dict(_snapshot_state_dict(state_dict))

        state, _ = env.reset(seed=1000 + worker_id)
        episode_reward = 0.0

        states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        dones: list[bool] = []

        while not stop_event.is_set():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = net(state_t)
                probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                action = int(dist.sample().item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(np.asarray(state, dtype=np.float32))
            actions.append(action)
            rewards.append(float(reward))
            dones.append(done)
            episode_reward += float(reward)
            state = next_state

            if done:
                try:
                    data_queue.put(("episode", episode_reward), timeout=0.1)
                except queue.Full:
                    pass
                state, _ = env.reset()
                episode_reward = 0.0

            if len(states) >= cfg.rollout_steps:
                next_state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    _, next_value = net(next_state_t)
                bootstrap = float(next_value.item())
                if dones[-1]:
                    bootstrap = 0.0

                returns: list[float] = []
                running = bootstrap
                for reward_i, done_i in zip(reversed(rewards), reversed(dones)):
                    if done_i:
                        running = 0.0
                    running = reward_i + cfg.gamma * running
                    returns.append(running)
                returns.reverse()

                payload = {
                    "states": np.asarray(states, dtype=np.float32),
                    "actions": np.asarray(actions, dtype=np.int64),
                    "returns": np.asarray(returns, dtype=np.float32),
                }
                try:
                    data_queue.put(("batch", payload), timeout=0.1)
                except queue.Full:
                    pass

                states.clear()
                actions.clear()
                rewards.clear()
                dones.clear()

            if stop_event.is_set():
                break

            net.load_state_dict(_snapshot_state_dict(state_dict))
    except Exception as exc:
        try:
            error_queue.put(
                {
                    "worker_id": worker_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                timeout=0.1,
            )
        except queue.Full:
            pass
    finally:
        if env is not None:
            env.close()


def run_a3c(config: A3CConfig | None = None) -> list[float]:
    cfg = config or A3CConfig()
    print("\n--- Starting A3C ---")

    mp_ctx = mp.get_context("spawn")

    probe_env = gym.make(cfg.env_name)
    obs_size = probe_env.observation_space.shape[0]
    n_actions = probe_env.action_space.n
    probe_env.close()

    net = ActorCritic(obs_size, n_actions, hidden_size=cfg.hidden_size)
    optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)

    manager = mp_ctx.Manager()
    shared_state = manager.dict({k: v.cpu() for k, v in net.state_dict().items()})
    data_queue = mp_ctx.Queue(maxsize=1024)
    error_queue = mp_ctx.Queue(maxsize=max(16, cfg.workers * 4))
    stop_event = mp_ctx.Event()

    workers: list[mp.Process] = []
    initial_state = {k: v.cpu() for k, v in net.state_dict().items()}
    shared_state.update(initial_state)

    for worker_id in range(cfg.workers):
        p = mp_ctx.Process(
            target=_worker_loop,
            args=(worker_id, cfg, shared_state, data_queue, error_queue, stop_event),
        )
        p.start()
        workers.append(p)

    rewards: list[float] = []
    worker_errors: list[dict[str, str]] = []
    try:
        while len(rewards) < cfg.episodes:
            while True:
                try:
                    worker_errors.append(error_queue.get_nowait())
                except queue.Empty:
                    break

            if worker_errors:
                first = worker_errors[0]
                raise RuntimeError(
                    f"A3C worker {first.get('worker_id')} failed: {first.get('error')}\n"
                    f"{first.get('traceback', '')}"
                )

            try:
                msg_type, payload = data_queue.get(timeout=1.0)
            except queue.Empty:
                if not any(p.is_alive() for p in workers):
                    raise RuntimeError("All A3C workers stopped before reaching target episodes.")
                continue

            if msg_type == "episode":
                rewards.append(float(payload))
                if len(rewards) % 50 == 0:
                    avg_reward = float(np.mean(rewards[-50:]))
                    print(f"Episode {len(rewards)}, Average Reward (last 50): {avg_reward:.2f}")
                continue

            if msg_type == "batch":
                states = torch.tensor(payload["states"], dtype=torch.float32)
                actions = torch.tensor(payload["actions"], dtype=torch.int64)
                returns_t = torch.tensor(payload["returns"], dtype=torch.float32)

                logits, values = net(states)
                values = values.squeeze(-1)

                probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                advantages = returns_t - values.detach()
                policy_loss = -(log_probs * advantages).mean()
                value_loss = nn.functional.mse_loss(values, returns_t)
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                shared_state.update({k: v.cpu() for k, v in net.state_dict().items()})
    finally:
        stop_event.set()
        for p in workers:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)

    return rewards
