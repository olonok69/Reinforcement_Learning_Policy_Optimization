# Benchmarks de RL Policy Optimization

Suite de benchmarks de reinforcement learning con múltiples algoritmos sobre `CartPole-v1`, con scripts independientes por método y un runner unificado de comparación.

## Alcance
Este repositorio incluye métodos de policy optimization:
- Policy Gradient (REINFORCE)
- A2C
- A3C
- PPO
- TRPO

## Documentación basada en libro
Explicaciones detalladas de algoritmos basadas en:
`doc/Deep Reinforcement Learning Hands-On_ Apply modern RL methods to practical problems of chatbots, robotics, discrete optimization, web automation, and more, 2nd Edition-Packt Publishing.pdf`

Ver índice: [doc/README.md](doc/README.md)

## Instalación
Usando `uv`:

```bash
uv sync
```

## Ejecutar un método (independiente)
Cada algoritmo tiene su propio script standalone:

```bash
uv run python policy_gradient_benchmark.py
uv run python a2c_benchmark.py
uv run python a3c_benchmark.py
uv run python ppo_benchmark.py
uv run python trpo_benchmark.py
```

## Ejecutar todos los métodos juntos
Orquestador unificado:

```bash
uv run python run_all_comparison.py
```

Ejecutar solo métodos seleccionados:

```bash
uv run python run_all_comparison.py --methods policy_gradient ppo trpo
```

Modo estricto (detener en el primer fallo):

```bash
uv run python run_all_comparison.py --strict
```

## Salidas
`run_all_comparison.py` genera:
- `outputs/comparison_results.json`
- `outputs/comparison_results.csv`
- `outputs/comparison_errors.json`

## Protocolo recomendado de experimentos
Para comparaciones justas y reproducibles entre métodos:

1. **Usar múltiples seeds**
   - Ejecutar cada método con al menos 3-5 seeds distintos.
   - Ejemplo: `--seed 1`, `--seed 2`, `--seed 3`.

2. **Mantener budgets alineados**
   - Igualar el budget de entrenamiento lo máximo posible (episodes o timesteps).
   - Reportar diferencias de budget cuando los métodos requieran estilos de rollout distintos.

3. **Reportar distribución, no una sola corrida**
   - Agregar `max_avg_reward_100` y `final_avg_reward_100` como media ± std entre seeds.
   - Mantener salidas crudas por corrida en `outputs/` para trazabilidad.

4. **Hacer un smoke-test rápido antes de corridas completas**
   - Primero correr un subconjunto de métodos para validar environment/dependencies.
   - Después lanzar la matriz completa de benchmark.

5. **Seguir fallos por separado**
   - Revisar `comparison_errors.json` y no eliminar métodos fallidos de los reportes.

## Agregar resultados multi-seed
Usa el script helper para agregar varias salidas JSON del benchmark en media/std por algoritmo:

```bash
uv run python scripts/aggregate_results.py --inputs outputs/*.json --output-json outputs/aggregate_summary.json --output-csv outputs/aggregate_summary.csv
```

También puedes pasar múltiples archivos explícitos o directorios:

```bash
uv run python scripts/aggregate_results.py --inputs runs/seed1 runs/seed2 outputs/comparison_results.json
```

Generar plots + reporte Markdown desde el resumen agregado:

```bash
uv run python scripts/generate_aggregate_report.py --input outputs/aggregate_summary.json --output-dir outputs/report --title "RL Policy Optimization Aggregate Report"
```

Reporte agregado en español:
- [outputs/report/aggregate_report_es.md](outputs/report/aggregate_report_es.md)

## Archivos principales
- `run_all_comparison.py`: runner unificado de benchmark multi-método
- `rl_comparison.py`: entrypoint de compatibilidad para comparación unificada
- `benchmarks/`: módulos reutilizables de algoritmos y utilidades compartidas de benchmark
- `doc/`: documentación de algoritmos y PDF fuente

## Notas
- TRPO usa la implementación de `sb3-contrib` (`benchmarks/trpo.py`).
- A3C usa multiprocessing y puede consumir CPU de forma significativa.
- Para comparaciones justas, mantener seeds y configuración del environment consistentes.
