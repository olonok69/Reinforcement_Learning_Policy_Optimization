# RL Benchmark Aggregate Report

Source data: `outputs/aggregate_summary.json`
Total algorithms: **5**

## Leaderboard (by MaxAvg100)

| Rank | Algo | Runs | MaxAvg100 | FinalAvg100 | Time(s) | Efficiency |
|---:|---|---:|---:|---:|---:|---:|
| 1 | TRPO | 1 | 500.00 | 500.00 | 144.51 | 3.460 |
| 2 | PPO | 1 | 68.80 | 55.25 | 21.69 | 3.172 |
| 3 | A2C | 1 | 59.48 | 58.86 | 27.56 | 2.159 |
| 4 | A3C | 1 | 18.90 | 11.46 | 26.03 | 0.726 |
| 5 | Policy Gradient | 1 | 17.20 | 11.45 | 26.11 | 0.659 |


## Plots

![MaxAvg100](plots/max_avg_reward_100.png)

![FinalAvg100](plots/final_avg_reward_100.png)

![ElapsedSeconds](plots/elapsed_seconds.png)

![PerformanceVsTime](plots/performance_vs_time.png)

## Key observations

- Top-3 by `max_avg_reward_100_mean`: **TRPO** (500.00), **PPO** (68.80), **A2C** (59.48)
- Bottom-3 by `max_avg_reward_100_mean`: **A2C** (59.48), **A3C** (18.90), **Policy Gradient** (17.20)
- Fastest method: **PPO** (21.69s)
- Slowest method: **TRPO** (144.51s)
- Median elapsed time across methods: **26.11s**

## Reproducibility note

Current summary appears to use single-run statistics (`runs=1` for each method), so standard deviations are zero. For robust comparisons, aggregate multiple seeds.

