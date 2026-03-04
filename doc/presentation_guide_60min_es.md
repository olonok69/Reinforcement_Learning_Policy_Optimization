# RL Policy Optimization Benchmarks — Guía de Presentación (60 minutos)

## 1) Objetivo de la sesión
Al final de esta charla, la audiencia debería entender:
- Qué hace esta aplicación
- Conceptos base de RL (MDP, return, exploration/exploitation)
- Qué significa **Policy Optimization** en la práctica
- Cómo cada algoritmo implementado se conecta con el código del repositorio
- Cómo ejecutar y comparar todos los métodos end-to-end

---

## 2) Agenda sugerida (60 minutos)

- **0–5 min**: framing del problema y overview de la aplicación
- **5–15 min**: fundamentos de RL (MDP, return, exploration/exploitation)
- **15–25 min**: objetivo de policy e intuición de policy gradient
- **25–40 min**: algoritmos implementados (walkthrough con links al código)
- **40–50 min**: flujo de demo en vivo (single method → all methods → aggregate report)
- **50–57 min**: interpretación de resultados y trade-offs
- **57–60 min**: conclusiones clave y Q&A

---

## 3) Qué es esta aplicación

Este repositorio es una aplicación de benchmarking de algoritmos de RL de policy optimization en `CartPole-v1`.

### Capacidades principales
- Ejecutar cada algoritmo de forma independiente (un script por método)
- Ejecutar todos los algoritmos desde un solo orquestador
- Guardar métricas estandarizadas (`episodes`, `elapsed_sec`, `max_avg_reward_100`, `final_avg_reward_100`)
- Agregar múltiples corridas/seeds
- Generar plots y un reporte en Markdown

### Entry points
- Orquestador unificado: [run_all_comparison.py](../run_all_comparison.py)
- Entry point de compatibilidad: [rl_comparison.py](../rl_comparison.py)

### Utilidades compartidas de benchmark
- [benchmarks/common.py](../benchmarks/common.py)

---

## 4) Conceptos de RL que deberías explicar

### 4.1 Framing MDP
Un problema de RL normalmente se modela como un Markov Decision Process (MDP):
- estado $s_t$
- acción $a_t$
- recompensa $r_t$
- dinámica de transición $P(s_{t+1}|s_t,a_t)$
- discount factor $\gamma$

El objetivo es maximizar el expected discounted return:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

### 4.2 ¿Qué es Policy Optimization?
Los métodos de Policy Optimization optimizan directamente los parámetros de política $\theta$ de $\pi_\theta(a|s)$ para maximizar expected return:

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

Idea típica de gradiente:

$$
\nabla_\theta J(\theta) \propto \mathbb{E}[\nabla_\theta\log \pi_\theta(a_t|s_t)\,\hat{A}_t]
$$

Donde $\hat{A}_t$ (advantage) indica si una acción fue mejor o peor que la expectativa base.

### 4.3 ¿Por qué optimizar la policy de forma directa?
Policy Optimization es útil cuando nos interesa aprender una distribución de acciones robusta, no solo una estimación de valor.

Intuición clave:
- Una policy es una distribución de probabilidad sobre acciones.
- Si una acción produce resultados mejores de lo esperado ($\hat{A}_t > 0$), aumentamos su probabilidad.
- Si una acción produce resultados peores de lo esperado ($\hat{A}_t < 0$), reducimos su probabilidad.

El término de gradiente $\nabla_\theta \log \pi_\theta(a_t|s_t)$ indica la dirección para aumentar probabilidad de la acción. Al multiplicarlo por $\hat{A}_t$, el update queda ponderado por "qué tan buena" fue esa acción.

### 4.4 Varianza, sesgo y estabilidad en práctica
En entrenamiento real, los policy gradients pueden ser ruidosos. Los métodos de este repo lo controlan con:
- **Baselines / critics** (A2C, A3C, PPO, TRPO) para reducir varianza.
- **Entropy regularization** para evitar policies deterministas demasiado pronto.
- **Trust constraints o clipping** (TRPO/PPO) para evitar saltos destructivos en parámetros.

### 4.5 Flujo on-policy (patrón común en estos métodos)
Los métodos implementados siguen mayormente un ciclo on-policy:
1. Generar trayectorias con la policy actual.
2. Estimar returns/advantages.
3. Optimizar policy (y value model cuando aplica).
4. Descartar trayectorias viejas y repetir con la policy actualizada.

Este enfoque suele ser estable, aunque menos eficiente en muestras que métodos con replay.

---

## 5) Mapa de algoritmos (con links al código)

### Policy Gradient (REINFORCE)
- Código: [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)
- Runner: [policy_gradient_benchmark.py](../policy_gradient_benchmark.py)
- Punto clave: returns Monte Carlo ponderan directamente gradientes de log-prob de la policy.
- Nota para presentación: enfatizarlo como estimador "puro"; simple, pero de mayor varianza.

### A2C (Advantage Actor-Critic)
- Código: [benchmarks/a2c.py](../benchmarks/a2c.py)
- Runner: [a2c_benchmark.py](../a2c_benchmark.py)
- Punto clave: actor + critic; advantage reduce la varianza frente a REINFORCE.
- Nota para presentación: actor decide y critic evalúa; esta división suele marcar el primer gran salto en estabilidad.

### A3C (Asynchronous Advantage Actor-Critic)
- Código: [benchmarks/a3c.py](../benchmarks/a3c.py)
- Runner: [a3c_benchmark.py](../a3c_benchmark.py)
- Punto clave: workers multi-proceso recolectan rollouts en forma asíncrona; un learner central actualiza el modelo compartido.
- Nota para presentación: destacar speedup en tiempo de pared y decorrelación por paralelismo.

### PPO (Proximal Policy Optimization)
- Código: [benchmarks/ppo.py](../benchmarks/ppo.py)
- Runner: [ppo_benchmark.py](../ppo_benchmark.py)
- Punto clave: objective con clipping para mejorar estabilidad de policy updates.
- Nota para presentación: suele ser el default práctico por balance entre simplicidad y estabilidad.

### TRPO (Trust Region Policy Optimization)
- Código: [benchmarks/trpo.py](../benchmarks/trpo.py)
- Runner: [trpo_benchmark.py](../trpo_benchmark.py)
- Punto clave: update con trust-region constraint (integrado con `sb3-contrib`).
- Nota para presentación: más principiado y conservador en updates, pero normalmente más costoso de optimizar.

---

## 6) Cómo ejecutar (comandos listos para demo)

### 6.1 Setup del environment
```bash
uv sync
```

### 6.2 Ejecutar un algoritmo
```bash
uv run python ppo_benchmark.py
```

### 6.3 Ejecutar todos los algoritmos
```bash
uv run python run_all_comparison.py
```

O subset seleccionado:
```bash
uv run python run_all_comparison.py --methods policy_gradient ppo trpo
```

### 6.4 Agregar salidas multi-seed
```bash
uv run python scripts/aggregate_results.py --inputs outputs/*.json --output-json outputs/aggregate_summary.json --output-csv outputs/aggregate_summary.csv
```

### 6.5 Generar plots + reporte
```bash
uv run python scripts/generate_aggregate_report.py --input outputs/aggregate_summary.json --output-dir outputs/report --title "RL Policy Optimization Aggregate Report"
```

Reporte generado:
- [outputs/report/aggregate_report.md](../outputs/report/aggregate_report.md)

---

## 7) Cómo presentar las métricas de salida

Métricas principales en esta app:
- `max_avg_reward_100`: mejor moving average en 100 episodes (capacidad pico)
- `final_avg_reward_100`: average de los últimos 100 episodes (estabilidad final)
- `elapsed_sec`: tiempo wall-clock (eficiencia de cómputo)
- `episodes`: total de episodes usados

Recomendación de interpretación:
- Comparar **peak** (`max_avg_reward_100`) para mejor comportamiento encontrado
- Comparar **final** (`final_avg_reward_100`) para calidad de convergencia
- Comparar **time** para costo práctico
- Usar múltiples seeds para confianza estadística

---

## 8) Narrativa sugerida para slides (concisa)

1. **¿Por qué una RL benchmark app?**
   - Necesidad de comparación apples-to-apples con mismo environment y mismas métricas.

2. **Intuición de policy optimization**
   - Mejorar directamente la política con returns y estimaciones de advantage.
   - Explicar por qué clipping/trust-region reduce cambios catastróficos de policy.

3. **Progresión de algoritmos**
   - REINFORCE → A2C/A3C → PPO/TRPO.

4. **Arquitectura de ingeniería**
   - Scripts independientes + orquestador + aggregation + report generation.

5. **Resultados y trade-offs**
   - Performance, estabilidad, tiempo de cómputo.

6. **Recomendación práctica**
   - Para baselines tipo producción: PPO/A2C, usando TRPO cuando se requiere update más conservador.

---

## 9) Preparación rápida para Q&A

- **¿Por qué un algoritmo obtiene mayor score pero tarda más?**
  - Diferencias en sample efficiency, estilo de optimización y overhead de cómputo.

- **¿Por qué necesitamos múltiples seeds?**
  - RL tiene alta varianza; una sola corrida puede ser engañosa.

- **¿Por qué mantener múltiples métodos de policy optimization?**
  - Ofrecen distintos trade-offs de estabilidad/eficiencia y opciones prácticas de despliegue.

---

## 10) Script de pizarra (5 minutos)

Úsalo como guion hablado rápido:

1. **Objetivo (30s)**
   - "Queremos una policy que mapee estados a acciones maximizando recompensa de largo plazo."

2. **Objeto central (45s)**
   - "Nuestra policy es $\pi_\theta(a|s)$, una distribución de probabilidad sobre acciones."
   - "Entrenar significa ajustar $\theta$ para que las mejores acciones sean más probables."

3. **Señal de aprendizaje (60s)**
   - "Tras rollouts, estimamos advantage $\hat{A}_t$."
   - "Si $\hat{A}_t>0$, subimos probabilidad de esa acción; si $\hat{A}_t<0$, la bajamos."
   - "Eso se refleja en $\nabla_\theta \log \pi_\theta(a_t|s_t)\,\hat{A}_t$."

4. **Progresión de algoritmos (90s)**
   - "REINFORCE: policy gradient Monte Carlo puro, simple pero ruidoso."
   - "A2C/A3C: agregan critic como baseline para reducir varianza; A3C añade workers asíncronos."
   - "PPO/TRPO: restringen tamaño del update (clip o trust region) para mayor estabilidad."

5. **Trade-off práctico (45s)**
   - "Updates más conservadores suelen mejorar estabilidad, pero pueden costar más cómputo/tiempo."
   - "PPO suele ser default práctico; TRPO para control más estricto del paso de policy."

6. **Cierre (30s)**
   - "Comparamos métodos con métricas compartidas: recompensa pico, recompensa final y tiempo."
   - "Siempre validar con múltiples seeds por la alta varianza en RL."

---

## 11) Documentación adicional del repo
- Guía principal del proyecto: [README.md](../README.md)
- Índice de docs de algoritmos: [doc/README.md](README.md)
