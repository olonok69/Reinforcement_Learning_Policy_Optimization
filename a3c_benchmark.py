from benchmarks.a3c import run_a3c
from benchmarks.common import run_timed, set_global_seed


if __name__ == "__main__":
    set_global_seed(42)
    _, result = run_timed(run_a3c, "A3C")
    print("\n--- Summary ---")
    print(f"{result.algo} took {result.elapsed_sec:.2f}s for {result.episodes} episodes.")
    print(f"{result.algo} Max Avg Reward (100): {result.max_avg_reward_100:.2f}")
    print(f"{result.algo} Final Avg Reward (100): {result.final_avg_reward_100:.2f}")
