from __future__ import annotations
"""Trust Region Policy Optimization (TRPO) benchmark wrapper.

This module delegates optimization details to `sb3-contrib` and exposes a
consistent benchmark interface used by the repository orchestrators.
"""

from dataclasses import dataclass
import importlib

import gymnasium as gym
import numpy as np

from benchmarks.common import record_policy_video


@dataclass
class TRPOConfig:
    """Configuration for TRPO benchmark execution."""

    env_name: str = "CartPole-v1"
    total_timesteps: int = 100_000
    record_video: bool = False
    video_dir: str = "videos/trpo"
    video_episodes: int = 3


def run_trpo(config: TRPOConfig | None = None) -> list[float]:
    """Train a TRPO agent and return monitored episode rewards."""

    cfg = config or TRPOConfig()
    print("\n--- Starting TRPO ---")

    try:
        trpo_module = importlib.import_module("sb3_contrib")
        TRPO = trpo_module.TRPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise ImportError(
            "TRPO benchmark requires sb3-contrib. Install with: pip install sb3-contrib"
        ) from exc

    rewards: list[float] = []

    class EpisodeRewardCallback(BaseCallback):
        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                ep = info.get("episode")
                if ep is not None and "r" in ep:
                    rewards.append(float(ep["r"]))
            return True

    env = Monitor(gym.make(cfg.env_name))

    model = TRPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
    )

    callback = EpisodeRewardCallback()
    model.learn(total_timesteps=cfg.total_timesteps, callback=callback)

    if cfg.record_video:
        def _policy(state: np.ndarray) -> int:
            action, _ = model.predict(state, deterministic=True)
            return int(action)

        record_policy_video(
            env_name=cfg.env_name,
            video_dir=cfg.video_dir,
            episodes=cfg.video_episodes,
            name_prefix="trpo",
            policy_fn=_policy,
        )

    env.close()
    return rewards
