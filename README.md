# RL Policy Optimization Benchmarks

Multi-algorithm reinforcement learning benchmark suite on `CartPole-v1`, with independent scripts per method and a unified comparison runner.

## Scope
This repository includes policy-optimization methods:
- Policy Gradient (REINFORCE)
- A2C
- A3C
- PPO
- TRPO

## Book-based documentation
Detailed algorithm writeups grounded in:
`doc/Deep Reinforcement Learning Hands-On_ Apply modern RL methods to practical problems of chatbots, robotics, discrete optimization, web automation, and more, 2nd Edition-Packt Publishing.pdf`

See the index: [doc/README.md](doc/README.md)

Spanish version of this file: [README_es.md](README_es.md)

## Install
Using `uv`:

```bash
uv sync
```

## Run single method (independent)
Each algorithm has its own standalone script:

```bash
uv run python policy_gradient_benchmark.py
uv run python a2c_benchmark.py
uv run python a3c_benchmark.py
uv run python ppo_benchmark.py
uv run python trpo_benchmark.py
```

## Run all methods together
Unified orchestrator:

```bash
uv run python run_all_comparison.py
```

Run selected methods only:

```bash
uv run python run_all_comparison.py --methods policy_gradient ppo trpo
```

Strict mode (stop on first failure):

```bash
uv run python run_all_comparison.py --strict
```

## Outputs
`run_all_comparison.py` writes:
- `outputs/comparison_results.json`
- `outputs/comparison_results.csv`
- `outputs/comparison_errors.json`

## Recommended experiment protocol
For fair and reproducible comparisons across methods:

1. **Use multiple seeds**
   - Run each method with at least 3-5 different seeds.
   - Example: `--seed 1`, `--seed 2`, `--seed 3`.

2. **Keep budgets aligned**
   - Match training budget as closely as possible (episodes or timesteps).
   - Report budget differences explicitly when methods require different rollout styles.

3. **Report distribution, not single runs**
   - Aggregate `max_avg_reward_100` and `final_avg_reward_100` as mean ± std across seeds.
   - Keep raw per-run outputs in `outputs/` for traceability.

4. **Use a quick smoke-test before full runs**
   - First run a small subset of methods to validate environment/dependencies.
   - Then launch the full benchmark matrix.

5. **Track failures separately**
   - Review `comparison_errors.json` and do not silently drop failed methods from reports.

## Aggregate multi-seed results
Use the helper script to aggregate several benchmark JSON outputs into per-algorithm mean/std:

```bash
uv run python scripts/aggregate_results.py --inputs outputs/*.json --output-json outputs/aggregate_summary.json --output-csv outputs/aggregate_summary.csv
```

You can also pass multiple explicit files or directories:

```bash
uv run python scripts/aggregate_results.py --inputs runs/seed1 runs/seed2 outputs/comparison_results.json
```

Generate plots + Markdown report from aggregated summary:

```bash
uv run python scripts/generate_aggregate_report.py --input outputs/aggregate_summary.json --output-dir outputs/report --title "RL Policy Optimization Aggregate Report"
```

## Main files
- `run_all_comparison.py`: unified multi-method benchmark runner
- `rl_comparison.py`: compatibility entrypoint to unified comparison
- `benchmarks/`: reusable algorithm modules and shared benchmark utilities
- `doc/`: algorithm documentation and source PDF

## Spanish documentation
- Root guide (ES): [README_es.md](README_es.md)
- Presentation guide (ES): [doc/presentation_guide_60min_es.md](doc/presentation_guide_60min_es.md)
- Algorithm docs (ES): [doc/es/README.md](doc/es/README.md)

## Notes
- TRPO uses `sb3-contrib` implementation (`benchmarks/trpo.py`).
- A3C uses multiprocessing and can consume significant CPU.
- For fair comparisons, keep seeds and environment settings consistent.
