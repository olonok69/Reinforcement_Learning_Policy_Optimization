import argparse
import json
from pathlib import Path
import traceback
import typing as tt

from benchmarks.common import BenchmarkResult, run_timed, save_results_csv, save_results_json, set_global_seed
from benchmarks.policy_gradient import run_policy_gradient
from benchmarks.a2c import run_a2c
from benchmarks.a3c import run_a3c
from benchmarks.ppo import run_ppo
from benchmarks.trpo import run_trpo


Runner = tt.Callable[[], list[float]]

METHODS: dict[str, tuple[str, Runner]] = {
    "policy_gradient": ("Policy Gradient", run_policy_gradient),
    "a2c": ("A2C", run_a2c),
    "a3c": ("A3C", run_a3c),
    "ppo": ("PPO", run_ppo),
    "trpo": ("TRPO", run_trpo),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified RL benchmark comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS.keys()),
        help="Methods to run. Default: all methods.",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on first failure instead of continuing with remaining methods.",
    )
    return parser.parse_args()


def validate_methods(methods: list[str]) -> list[str]:
    unknown = [name for name in methods if name not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {list(METHODS.keys())}")
    return methods


def print_summary(results: list[BenchmarkResult], errors: list[dict]) -> None:
    print("\n--- Unified Summary ---")
    if results:
        for r in sorted(results, key=lambda x: x.max_avg_reward_100, reverse=True):
            print(
                f"{r.algo:16} | episodes={r.episodes:4d} | "
                f"max100={r.max_avg_reward_100:7.2f} | final100={r.final_avg_reward_100:7.2f} | "
                f"time={r.elapsed_sec:8.2f}s"
            )
    else:
        print("No successful runs.")

    if errors:
        print("\nFailed methods:")
        for err in errors:
            print(f"- {err['method']}: {err['error']}")


def main() -> None:
    args = parse_args()
    selected_methods = validate_methods(args.methods)
    set_global_seed(args.seed)

    results: list[BenchmarkResult] = []
    errors: list[dict] = []

    for method_key in selected_methods:
        display_name, runner = METHODS[method_key]
        print(f"\n=== Running {display_name} ({method_key}) ===")
        try:
            _, result = run_timed(runner, display_name)
            results.append(result)
        except Exception as exc:
            errors.append(
                {
                    "method": method_key,
                    "display_name": display_name,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[ERROR] {display_name} failed: {exc}")
            if args.strict:
                break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "comparison_results.json"
    csv_path = output_dir / "comparison_results.csv"
    errors_path = output_dir / "comparison_errors.json"

    save_results_json(results, json_path)
    save_results_csv(results, csv_path)
    errors_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")

    print_summary(results, errors)
    print(f"\nSaved results JSON: {json_path}")
    print(f"Saved results CSV:  {csv_path}")
    print(f"Saved errors JSON:  {errors_path}")


if __name__ == "__main__":
    main()
