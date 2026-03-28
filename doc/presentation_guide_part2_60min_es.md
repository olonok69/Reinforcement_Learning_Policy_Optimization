# RL Policy Optimization — Guía de Presentación Parte 2 (60 minutos)

## Alcance de la Parte 2
Esta sesión cubre los algoritmos restantes después de REINFORCE:
- A2C
- A3C
- PPO
- TRPO

Enfoque:
- por qué estos métodos mejoran estabilidad/eficiencia frente a REINFORCE
- cómo mapear cada método al código del repositorio
- cómo comparar métodos con un flujo de benchmark reproducible

---

## 1) Objetivo de la sesión
Al terminar esta Parte 2, la audiencia debería entender:
- el rol de critic baselines para reducir varianza
- por qué la recolección asíncrona puede ayudar (A3C)
- por qué clipping/trust-region estabiliza entrenamiento (PPO/TRPO)
- cómo ejecutar y comparar todos los métodos en este repositorio

---

## 2) Agenda sugerida (60 minutos)

- **0–6 min**: recap rápido de Parte 1 (limitaciones de REINFORCE)
- **6–20 min**: A2C (actor + critic)
- **20–30 min**: A3C (workers asíncronos)
- **30–43 min**: PPO (surrogate con clipping + GAE)
- **43–50 min**: TRPO (concepto trust-region + integración de librería)
- **50–57 min**: flujo de ejecución/comparación de benchmark
- **57–60 min**: resumen de recomendaciones + Q&A

---

## 3) A2C (Advantage Actor-Critic)

### Intuición principal
REINFORCE usa returns completos directos y puede ser ruidoso.
A2C introduce baseline de valor $V(s)$ y usa advantage:

$$
\hat{A}_t = R_t - V(s_t)
$$

Esto normalmente reduce varianza del gradiente.

### Mapa de código
- Algoritmo: [benchmarks/a2c.py](../benchmarks/a2c.py)
- Runner standalone: [a2c_benchmark.py](../a2c_benchmark.py)

### Qué señalar en el código
- shared backbone + policy/value heads
- policy loss + value loss + entropy term
- cálculo de returns y advantage por episodio

---

## 4) A3C (Asynchronous Advantage Actor-Critic)

### Intuición principal
A3C mantiene estructura actor-critic pero recolecta experiencia con múltiples workers en paralelo.

Beneficios:
- decorrelación de datos
- mejora de tiempo de pared en CPU

Trade-off:
- más complejidad de sistema (procesos, colas, sincronización)

### Mapa de código
- Algoritmo: [benchmarks/a3c.py](../benchmarks/a3c.py)
- Runner standalone: [a3c_benchmark.py](../a3c_benchmark.py)

### Qué señalar en el código
- worker loop con episodios y mini-rollouts
- learner consumiendo batches desde cola
- patrón de refresco de parámetros compartidos

---

## 5) PPO (Proximal Policy Optimization)

### Intuición principal
PPO evita saltos grandes de policy con objective clippeado:

$$
L^{clip}(\theta)=\mathbb{E}[\min(r_t(\theta)\hat{A}_t,\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)]
$$

Suele ser default práctico por balance entre estabilidad y simplicidad.

### Mapa de código
- Algoritmo: [benchmarks/ppo.py](../benchmarks/ppo.py)
- Runner standalone: [ppo_benchmark.py](../ppo_benchmark.py)

### Qué señalar en el código
- recolección de rollouts
- cálculo de GAE
- updates por minibatch y múltiples epochs con clipping

---

## 6) TRPO (Trust Region Policy Optimization)

### Intuición principal
TRPO restringe updates para mantener la policy en una trust region (típicamente con constraint KL).

Pros:
- updates conservadores y principiados

Contras:
- maquinaria de optimización más pesada

### Mapa de código
- Wrapper de integración: [benchmarks/trpo.py](../benchmarks/trpo.py)
- Runner standalone: [trpo_benchmark.py](../trpo_benchmark.py)

### Qué señalar en el código
- dependencia de `sb3-contrib`
- captura de recompensas por callback
- consistencia de interfaz de benchmark respecto a otros métodos

---

## 7) Narrativa de comparación lado a lado

| Método | Idea principal | Fortaleza | Trade-off común |
|---|---|---|---|
| A2C | baseline critic + advantage | Más estable que REINFORCE | Sigue siendo on-policy y menos sample-efficient |
| A3C | actor-critic asíncrono multi-worker | Mejor comportamiento en tiempo de pared (CPU) | Complejidad de multiprocessing |
| PPO | clipping + updates por minibatch repetidos | Default práctico fuerte | Sensible a hiperparámetros |
| TRPO | updates restringidos por trust-region | Mejora conservadora estable | Mayor costo computacional |

---

## 8) Comandos de demo (Parte 2)

### Ejecutar cada método
```bash
uv run python a2c_benchmark.py
uv run python a3c_benchmark.py
uv run python ppo_benchmark.py
uv run python trpo_benchmark.py
```

### Ejecutar todos los métodos con un comando
```bash
uv run python run_all_comparison.py --methods policy_gradient a2c a3c ppo trpo
```

### Ejecutar subset con videos
```bash
uv run python run_all_comparison.py --methods a2c ppo trpo --record-video --video-dir videos --video-episodes 2
```

### Ejemplo de alineación de budgets
```bash
uv run python run_all_comparison.py --methods a2c a3c ppo trpo --a2c-episodes 600 --a3c-episodes 600 --ppo-episodes 700 --trpo-timesteps 200000
```

---

## 9) Reportes e interpretación

Métricas principales del comparison runner:
- `max_avg_reward_100`
- `final_avg_reward_100`
- `elapsed_sec`
- `episodes`

Interpretación recomendada:
- comparar capacidad pico (`max_avg_reward_100`)
- comparar estabilidad final (`final_avg_reward_100`)
- comparar costo práctico (`elapsed_sec`)
- repetir en múltiples seeds y agregar resultados

Pipeline de agregación:
- [../scripts/aggregate_results.py](../scripts/aggregate_results.py)
- [../scripts/generate_aggregate_report.py](../scripts/generate_aggregate_report.py)

---

## 10) Cierre y recomendaciones

Orden práctico sugerido para baseline:
1. Empezar por PPO.
2. Usar A2C para un baseline actor-critic más simple.
3. Usar A3C cuando interese paralelismo CPU en recolección.
4. Usar TRPO si se necesita control más conservador del update y hay presupuesto de cómputo.

Para bases conceptuales sólidas, siempre anclar primero en Parte 1 (policy optimization + Monte Carlo + REINFORCE) antes del deep-dive de variantes.
