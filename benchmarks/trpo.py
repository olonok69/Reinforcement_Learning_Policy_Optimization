from __future__ import annotations

from dataclasses import dataclass
import importlib

import gymnasium as gym


@dataclass
class TRPOConfig:
    env_name: str = "CartPole-v1"
    total_timesteps: int = 100_000


def run_trpo(config: TRPOConfig | None = None) -> list[float]:
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

    env.close()
    return rewards
