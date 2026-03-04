from benchmarks.common import run_timed, set_global_seed
from benchmarks.trpo import run_trpo


if __name__ == "__main__":
    set_global_seed(42)
    _, result = run_timed(run_trpo, "TRPO")
    print("\n--- Summary ---")
    print(f"{result.algo} took {result.elapsed_sec:.2f}s for {result.episodes} episodes.")
    print(f"{result.algo} Max Avg Reward (100): {result.max_avg_reward_100:.2f}")
    print(f"{result.algo} Final Avg Reward (100): {result.final_avg_reward_100:.2f}")
