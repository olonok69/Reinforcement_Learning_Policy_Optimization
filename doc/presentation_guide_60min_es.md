# RL Policy Optimization — Guía de Presentación Parte 1 (60 minutos)

## Alcance de la Parte 1
Esta primera sesión está diseñada con mucho detalle para una audiencia de nivel inicial.

Cubre:
- Fundamentos de RL necesarios antes de policy optimization
- Por qué aparece Monte Carlo en REINFORCE
- Intuición del policy gradient theorem
- REINFORCE end-to-end (matemática + mapeo a código)
- Cómo ejecutar y explicar el código del repositorio para REINFORCE

No cubre en profundidad A2C/A3C/PPO/TRPO. Eso se ve en la Parte 2.

---

## 1) Objetivo de la sesión
Al terminar esta Parte 1, la audiencia debería poder explicar:
- Qué es una policy y por qué optimizarla directamente
- Qué significa return en tareas episódicas
- Por qué los returns Monte Carlo son insesgados pero con alta varianza
- Cómo REINFORCE actualiza los parámetros de la policy
- Cómo se conecta la teoría con el código del repositorio

---

## 2) Agenda sugerida (60 minutos)

- **0–8 min**: framing del problema + qué hace esta app
- **8–20 min**: fundamentos de RL (MDP, trayectoria, return)
- **20–33 min**: policy optimization e intuición de policy gradient
- **33–48 min**: walkthrough detallado de REINFORCE (ecuaciones + pseudo-código + links a código)
- **48–55 min**: demo en vivo (single method + compare mode)
- **55–60 min**: Q&A + transición a Parte 2

---

## 3) Qué es esta aplicación

Este repositorio es una app de benchmarking de métodos de policy optimization sobre `CartPole-v1`.

### Capacidades principales
- Scripts standalone por algoritmo
- Orquestador unificado para comparación de algoritmos
- Métricas estandarizadas y salidas reproducibles
- Agregación de corridas y generación de reportes

### Entry points
- Orquestador unificado: [run_all_comparison.py](../run_all_comparison.py)
- Runner REINFORCE standalone: [policy_gradient_benchmark.py](../policy_gradient_benchmark.py)

### Módulo central de REINFORCE para esta sesión
- [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)

---

## 4) Fundamentos de RL (enfoque beginner)

### 4.1 Bloques de un MDP
Una tarea de RL suele modelarse como un Markov Decision Process (MDP):
- estado: $s_t$
- acción: $a_t$
- recompensa: $r_t$
- dinámica de transición: $P(s_{t+1}|s_t,a_t)$
- factor de descuento: $\gamma \in [0,1]$

### 4.2 Trayectoria y return
Un episodio (trayectoria) es:

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\dots)
$$

Return descontado desde tiempo $t$:

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}
$$

Intuición low-level:
- Las recompensas inmediatas pesan más cuando $\gamma<1$.
- El return es la señal de entrenamiento que indica si el comportamiento muestreado fue bueno.

### 4.3 ¿Por qué policies estocásticas?
En policy optimization modelamos:

$$
\pi_\theta(a|s)
$$

como distribución de probabilidad sobre acciones. Esto ayuda a:
- explorar durante entrenamiento
- tener gradientes suaves para optimizar
- aprender preferencias de acción, no solo decisiones rígidas

---

## 5) Policy optimization desde primeros principios

### 5.1 Objetivo
Maximizamos el retorno esperado de trayectorias:

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

### 5.2 Truco de derivada logarítmica (visión general)
Diferenciar directamente la probabilidad de una trayectoria es difícil.
Policy gradient usa:

$$
\nabla_\theta p_\theta(\tau)=p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)
$$

y eso permite estimar el gradiente usando trayectorias muestreadas.

### 5.3 Forma del estimador de policy gradient
Una forma común:

$$
\nabla_\theta J(\theta)\propto\mathbb{E}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)\,G_t\right]
$$

Interpretación para principiantes:
- $\nabla_\theta\log\pi_\theta(a_t|s_t)$: “dirección para aumentar la probabilidad de la acción tomada”
- $G_t$: “qué tan bueno resultó ese outcome”
- Multiplicar ambos refuerza acciones asociadas a mejores retornos.

---

## 6) Monte Carlo en REINFORCE

### 6.1 Qué significa Monte Carlo aquí
REINFORCE espera al final del episodio y calcula returns completos con recompensas observadas.
No usa bootstrap de un value target en su forma base.

### 6.2 Por qué es útil
- Estimación insesgada del return para la policy muestreada
- Implementación simple

### 6.3 Por qué es difícil
- Updates con alta varianza
- Entrenamiento potencialmente inestable y más lento

### 6.4 Ejemplo numérico pequeño
Si un episodio tiene recompensas: $[1,1,1]$ con $\gamma=0.9$:

$$
G_0=1+0.9+0.9^2=2.71,
\quad G_1=1+0.9=1.9,
\quad G_2=1
$$

La acción temprana pesa más porque impacta más recompensas futuras.

---

## 7) Algoritmo REINFORCE (paso a paso)

1. Reset del environment y muestreo de episodio con policy actual.
2. Guardar log-probabilities por paso y recompensas.
3. Calcular returns descontados por timestep.
4. (Opcional) normalizar returns para estabilizar escala.
5. Calcular pérdida:

$$
\mathcal{L}_{policy}=-\sum_t\log\pi_\theta(a_t|s_t)\,G_t
$$

6. Backpropagation y update con Adam.
7. Repetir por muchos episodios.

---

## 8) Mapeo a código para Parte 1

### Implementación REINFORCE
- Config/dataclass: [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)
- Red de policy (`PolicyNetwork`): [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)
- Helper de returns (`_discounted_returns`): [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)
- Training loop (`run_policy_gradient`): [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)

### Runner CLI para demo
- [policy_gradient_benchmark.py](../policy_gradient_benchmark.py)

### Entrypoint de comparación unificada
- [run_all_comparison.py](../run_all_comparison.py)

---

## 9) Comandos de demo (Parte 1)

### Setup
```bash
uv sync
```

### Ejecutar solo REINFORCE
```bash
uv run python policy_gradient_benchmark.py
```

### REINFORCE con hiperparámetros
```bash
uv run python policy_gradient_benchmark.py --episodes 900 --gamma 0.99 --learning-rate 0.001
```

### REINFORCE con video de evaluación
```bash
uv run python policy_gradient_benchmark.py --record-video --video-dir videos/policy_gradient --video-episodes 2
```

### Desde orquestador unificado (solo REINFORCE)
```bash
uv run python run_all_comparison.py --methods policy_gradient --policy-gradient-episodes 900
```

---

## 10) Notas didácticas para audiencia low-level

Secuencia sugerida en slides/pizarra:

1. Definir policy como probabilidades sobre acciones.
2. Explicar una trayectoria muestreada como “evidencia”.
3. Mostrar return descontado y por qué acciones tempranas afectan más el futuro.
4. Mostrar policy gradient como “subir probabilidad de acciones con mejor outcome”.
5. Aclarar trade-off Monte Carlo: simple + insesgado, pero ruidoso.

Confusiones comunes a resolver explícitamente:
- “¿Esto es supervised learning?” → No; no hay labels, hay rewards diferidas del entorno.
- “¿Por qué no elegir siempre argmax?” → La estocasticidad ayuda exploración y aprendizaje por gradiente.
- “¿Por qué normalizar returns?” → Para estabilizar magnitud de updates.

---

## 11) Puente a Parte 2

Parte 2 responde la siguiente pregunta natural tras REINFORCE:
- ¿Cómo reducir varianza y mejorar estabilidad/rendimiento?

Ahí entran:
- A2C / A3C (critic baseline + paralelismo)
- PPO (updates con clipping)
- TRPO (updates con trust-region)

Guía Parte 2:
- [presentation_guide_part2_60min_es.md](presentation_guide_part2_60min_es.md)

---

## 12) Lecturas externas sugeridas (refuerzo conceptual)

Úsalas como referencias opcionales para profundizar conceptos:
- https://cameronrwolfe.substack.com/p/policy-gradients-the-foundation-of
- https://www.linkedin.com/pulse/policy-gradient-theorem-continuous-tasks-rl-abram-george/
- https://jonathan-hui.medium.com/rl-policy-gradients-explained-9b13b688b146

Estas referencias complementan la presentación; el código del repositorio sigue siendo la fuente principal para detalles de implementación.
