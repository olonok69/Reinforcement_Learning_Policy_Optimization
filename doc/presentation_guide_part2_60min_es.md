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
A2C introduce baseline de valor `V(s)` y usa advantage:

```
Aₜ = Rₜ - V(sₜ)
```

Esto normalmente reduce varianza del gradiente.

### Mapa de código
- Algoritmo: [benchmarks/a2c.py](../benchmarks/a2c.py)
- Runner standalone: [a2c_benchmark.py](../a2c_benchmark.py)

### Qué señalar en el código
- shared backbone + policy/value heads
- policy loss + value loss + entropy term
- cálculo de returns y advantage por episodio

### Cómo funciona A2C paso a paso

```
1. Recolectar un episodio completo con la policy actual
2. Calcular returns descontados Monte Carlo Gₜ para cada paso
3. Obtener V(sₜ) de la cabeza critic para cada estado
4. Calcular advantage: Aₜ = Gₜ - V(sₜ)
5. Calcular loss combinado:
     L = -Σ log π(a|s) · A        (policy — ponderado por advantage)
       + value_coef · (V(s) - G)²  (critic — aprender a predecir returns)
       - entropy_coef · H(π)       (entropía — prevenir colapso)
6. Backpropagate y actualizar ambas cabezas conjuntamente
7. Repetir por muchos episodios
```

### Correspondencia teoría-código

| Concepto teórico | Código en `benchmarks/a2c.py` |
|-----------------|------------------------------|
| `π(a\|s)` — actor | `ActorCritic.policy_head` → `Categorical(logits)` |
| `V(s)` — critic | `ActorCritic.value_head` → escalar |
| `A = G - V(s)` — advantage | `advantage = returns - values.detach()` |
| Policy loss | `-(log_probs * advantage).sum()` |
| Value loss | `F.mse_loss(values, returns)` |
| Entropy bonus | `dist.entropy().mean()` |
| Loss combinado | `policy_loss + 0.5*value_loss - 0.01*entropy` |

### Qué arregla A2C vs REINFORCE
- **Menor varianza** — el baseline de advantage elimina ruido de returns crudos
- **Mejor asignación de crédito** — solo acciones sobre el promedio se refuerzan
- **Protección de exploración** — bonus de entropía previene colapso prematuro
- **Sigue siendo on-policy** — cada trayectoria se usa una vez y se descarta

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

### Arquitectura en detalle

Cada worker: ejecuta su propia copia del entorno → recolecta `rollout_steps` transiciones → calcula advantages → envía batch a la cola compartida.

Learner: desencola batches → aplica loss combinado (igual que A2C) → actualiza parámetros compartidos del modelo.

Los workers refrescan periódicamente sus pesos locales desde el modelo compartido.

### Config por defecto
`workers=4, rollout_steps=5, γ=0.99, lr=1e-3, value_coef=0.5, entropy_coef=0.01`

### A2C vs A3C
- **A2C** — updates síncronos, un solo proceso, más fácil de debuggear y reproducir
- **A3C** — workers asíncronos, multiprocessing, mejor velocidad en CPUs multi-core
- **Trade-off**: multiprocessing agrega complejidad de ingeniería (colas, sincronización, manejo de errores)

---

## 5) PPO (Proximal Policy Optimization)

### Intuición principal
PPO evita saltos grandes de policy con objective clippeado:

```
L_clip(θ) = E[ min( rₜ(θ)·Aₜ , clip(rₜ(θ), 1-ε, 1+ε)·Aₜ ) ]

donde rₜ(θ) = π_new(aₜ|sₜ) / π_old(aₜ|sₜ)
```

Suele ser default práctico por balance entre estabilidad y simplicidad.

### Mapa de código
- Algoritmo: [benchmarks/ppo.py](../benchmarks/ppo.py)
- Runner standalone: [ppo_benchmark.py](../ppo_benchmark.py)

### Qué señalar en el código
- recolección de rollouts (1024 pasos)
- cálculo de GAE (`_compute_gae()`)
- updates por minibatch y múltiples epochs con clipping

### El mecanismo de clipping explicado

El ratio de probabilidad mide cuánto cambió la policy:

```
r(θ) = π_new(a|s) / π_old(a|s)
```

- `r = 1.0` → policy sin cambios
- `r = 1.5` → acción 50% más probable bajo nueva policy
- `r = 0.5` → acción 50% menos probable

Clipear a `[1-ε, 1+ε]` (default `[0.8, 1.2]`) limita cuánto puede empujar un solo update:

```python
# benchmarks/ppo.py — el clipping central
ratio = torch.exp(new_log_probs - old_log_probs)
surr1 = ratio * advantage
surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantage
policy_loss = -torch.min(surr1, surr2).mean()
```

### GAE (Generalized Advantage Estimation)

GAE mezcla errores TD multi-paso con parámetro `λ` (default 0.95):

```
λ=0:    TD puro de 1 paso (baja varianza, alto sesgo)
λ=1:    Monte Carlo completo (alta varianza, sin sesgo)
λ=0.95: punto óptimo — mayormente MC pero suavizado por TD
```

### Flujo de entrenamiento
1. Recolectar rollout (1024 pasos) con policy actual
2. Calcular advantages con GAE
3. Normalizar advantages
4. Ejecutar 4 epochs por minibatch (shuffle + split en batches de 64) con loss clipeado
5. Descartar rollout, repetir

### Config por defecto
`rollout_steps=1024, update_epochs=4, minibatch_size=64, clip_eps=0.2, gae_lambda=0.95, lr=3e-4, value_coef=0.5, entropy_coef=0.01`

### Por qué PPO es el default práctico
- **Estable** — clipping previene updates catastróficos
- **Simple** — no requiere optimización restringida (a diferencia de TRPO)
- **Reutiliza datos** — múltiples epochs por rollout (más sample-efficient que A2C)
- **RLHF** — usado para finetunear ChatGPT, LLaMA, y otros LLMs

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

### La restricción de trust region

```
maximizar  E[ r(θ) · A ]
sujeto a   KL( π_old || π_new ) ≤ δ
```

KL divergence mide qué tan diferentes son dos distribuciones de probabilidad. La restricción dice: "mejora la policy, pero no la cambies demasiado respecto a la actual."

TRPO resuelve esto exactamente usando gradiente conjugado + line search. Teóricamente riguroso pero computacionalmente costoso por update.

### TRPO vs PPO
- **TRPO** — resuelve la optimización restringida exactamente. Mejora monótona garantizada. Costoso.
- **PPO** — aproxima la misma idea con clipping simple. Mucho más barato, casi igual de estable, más fácil de implementar.
- PPO fue diseñado como alternativa más simple a TRPO. En la práctica, PPO es preferido para la mayoría de tareas.

### Cuándo usar TRPO
- Sistemas safety-critical o robótica donde updates conservadores son esenciales
- Cuando se necesita garantía teórica de mejora monótona
- Cuando el presupuesto de cómputo permite el costo extra por update

### Código: benchmarks/trpo.py
Este repo usa la implementación de TRPO de `sb3-contrib` envuelta para compatibilidad de benchmark. Como sb3 maneja el training loop internamente, las recompensas se capturan por callback para consistencia con los otros métodos.

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