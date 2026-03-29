# RL Policy Optimization — Guía de Presentación Parte 1 (60 minutos)

## Alcance de la Parte 1
Esta primera sesión está diseñada con mucho detalle para una audiencia de nivel inicial.

Cubre:
- Fundamentos de RL necesarios antes de policy optimization
- Por qué aparece Monte Carlo en REINFORCE
- Intuición del policy gradient theorem y roadmap de derivación completa
- Funciones de valor, función de ventaja y su rol en reducir varianza
- Variantes del policy gradient (reward-to-go, baseline, vanilla PG)
- REINFORCE end-to-end (matemática + mapeo a código)
- Conexión con RLHF y finetuning moderno de LLMs (PPO)
- Cómo ejecutar y explicar el código del repositorio para REINFORCE

No cubre en profundidad A2C/A3C/PPO/TRPO. Eso se ve en la Parte 2.

**Referencia clave**: [Cameron R. Wolfe — Policy Gradients: The Foundation of RLHF](https://cameronrwolfe.substack.com/p/policy-gradients-the-foundation-of)

---

## 1) Objetivo de la sesión
Al terminar esta Parte 1, la audiencia debería poder explicar:
- Qué es una policy y por qué optimizarla directamente
- Qué significa return en tareas episódicas
- Por qué los returns Monte Carlo son insesgados pero con alta varianza
- Cómo el log-derivative trick hace computables los policy gradients
- Cómo REINFORCE actualiza los parámetros de la policy
- Qué es la función de ventaja y por qué importa para A2C/PPO
- Cómo las cuatro variantes del policy gradient reducen varianza progresivamente
- Cómo se conecta la teoría con el código del repositorio
- Por qué los policy gradients son la base de RLHF para LLMs

---

## 2) Agenda sugerida (60 minutos)

- **0–8 min**: framing del problema + qué hace esta app
- **8–18 min**: fundamentos de RL (MDP, trayectoria, return)
- **18–28 min**: objetivo de policy optimization + log-derivative trick
- **28–38 min**: policy gradient theorem + funciones de valor/ventaja
- **38–48 min**: walkthrough detallado de REINFORCE + variantes del policy gradient
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
- **estado** `sₜ` — lo que el agente observa
- **acción** `aₜ` — lo que el agente hace
- **recompensa** `rₜ` — feedback inmediato del entorno
- **dinámica de transición** `P(sₜ₊₁ | sₜ, aₜ)` — cómo cambia el mundo
- **factor de descuento** `γ ∈ [0, 1]` — cuánto valoramos recompensas futuras vs inmediatas

### 4.2 Trayectoria y return
Un episodio (trayectoria) es:

```
τ = (s₀, a₀, r₀, s₁, a₁, r₁, ...)
```

Return descontado desde tiempo `t`:

```
Gₜ = rₜ + γ · rₜ₊₁ + γ² · rₜ₊₂ + ...
   = Σₖ γᵏ · rₜ₊ₖ
```

Intuición low-level:
- Las recompensas inmediatas pesan más cuando `γ < 1`.
- El return es la señal de entrenamiento que indica si el comportamiento muestreado fue bueno.

Dos tipos de return (del artículo de Wolfe):
- **Horizonte finito sin descuento**: suma de todas las recompensas en un episodio de largo fijo
- **Horizonte infinito con descuento**: suma con `γ` para asegurar convergencia

### 4.3 ¿Por qué policies estocásticas?
En policy optimization modelamos:

```
πθ(a | s)  =  P(acción = a | estado = s)
```

como distribución de probabilidad sobre acciones. Esto ayuda a:
- exploración durante entrenamiento
- gradientes suaves para optimización
- aprender preferencias, no solo decisiones duras

---

## 5) Value-Based vs Policy-Based

Dos enfoques fundamentalmente diferentes para RL:

| Aspecto | Value-Based (DQN) | Policy-Based (Esta Serie) |
|---------|-------------------|--------------------------|
| Estrategia | Aprender Q(s,a), elegir argmax | Optimizar π(a\|s) directamente |
| Policy | Implícita (derivada de Q) | Explícita (red neuronal genera probabilidades) |
| Acciones | Solo discretas (fácilmente) | Continuas y discretas |
| Fundamento | Ecuación de Bellman | Policy gradient theorem |
| Usado en RLHF | No | Sí — PPO es policy optimization |

**Analogía para beginners**:
- Value-based: calificar cada plato del menú y siempre elegir el mejor puntuado
- Policy-based: desarrollar un gusto personal — aprendes directamente qué platos prefieres, sin calificar cada uno

---

## 6) Policy optimization desde primeros principios

### 6.1 Objetivo
Maximizamos el return esperado de trayectorias:

```
J(θ) = E_τ~πθ [ R(τ) ]
```

En palabras: la recompensa total promedio que obtenemos al muestrear trayectorias de la policy `π`.

### 6.2 Gradient ascent
Para maximizar `J(θ)`, usamos gradient ascent:

```
θ  ←  θ + α · ∇θ J(θ)
```

En cada paso:
1. Calcular gradiente del objetivo respecto a los parámetros actuales
2. Multiplicar por learning rate `α`
3. Mover parámetros cuesta arriba (sumar, no restar — ascent, no descent)

### 6.3 La parte difícil
Calcular `∇J(θ)` es difícil porque el gradiente involucra `P(τ|θ)` — la probabilidad de una trayectoria — que depende tanto de la policy como de la dinámica desconocida del entorno.

**Solución**: El log-derivative trick (siguiente sección).

---

## 7) El log-derivative trick (paso a paso)

Este es el insight matemático clave que hace prácticos los policy gradients.

### Paso 1: El problema
Necesitamos:

```
∇J(θ) = ∇ Σ P(τ|θ) · R(τ)
```

¡Pero `P(τ|θ)` incluye dinámicas del entorno que no conocemos!

### Paso 2: La identidad de cálculo
Del cálculo básico, la derivada de log es:

```
∇ log f(x)  =  ∇f(x) / f(x)
```

Reorganizando:

```
∇f(x)  =  f(x) · ∇ log f(x)
```

### Paso 3: Aplicar a ∇P(τ|θ)

```
∇P(τ|θ)  =  P(τ|θ) · ∇log P(τ|θ)
```

¡Ahora `P(τ|θ)` reaparece como multiplicador → la suma se convierte en una **esperanza** que podemos muestrear!

### Paso 4: El entorno se cancela
La probabilidad de trayectoria es un producto:

```
P(τ|θ) = P(s₀) · Π πθ(aₜ|sₜ) · P(sₜ₊₁|sₜ,aₜ)
```

Tomar logaritmo convierte productos en sumas:

```
log P(τ|θ) = log P(s₀) + Σ log πθ(aₜ|sₜ) + Σ log P(sₜ₊₁|sₜ,aₜ)
```

**Al tomar ∇ respecto a θ, ¡solo `πθ(a|s)` depende de θ!**
Los términos del entorno `P(s₀)` y `P(s'|s,a)` desaparecen porque su gradiente respecto a θ es cero.

### Paso 5: El resultado

```
∇J(θ) = E[ Σₜ ∇log πθ(aₜ|sₜ) · Gₜ ]
```

Solo necesitamos dos cosas que SÍ podemos calcular:
1. Gradiente de log-policy (de la red neuronal — autograd lo maneja)
2. Return `G` (de episodios muestreados — solo sumar recompensas)

**¡No se necesita modelo del entorno! Esto es model-free RL.**

---

## 8) Policy gradient theorem — la ecuación central

### El estimador (versión práctica)

```
∇J(θ)  ≈  (1/N) Σᵢ [ Σₜ ∇log πθ(aₜ|sₜ) · Gₜ ]
```

Muestreamos `N` trayectorias, calculamos la expresión para cada una, y promediamos.

### Desglose — palabra por palabra

| Término | Significado |
|---------|-----------|
| `∇log π(a\|s)` | "Dirección para aumentar la probabilidad de la acción elegida `a` en estado `s`" |
| `Gₜ` (Return) | "¿Qué tan bueno fue ese resultado muestreado?" |
| Multiplicar | "Los buenos outcomes REFUERZAN sus acciones, los malos las disminuyen" |
| `(1/N) Σ` | "Promediar sobre N trayectorias muestreadas para estimar la esperanza" |

### Intuición
- Si una acción llevó a **alto return** → aumentar su probabilidad
- Si una acción llevó a **bajo return** → disminuir su probabilidad
- El gradiente apunta en la dirección que hace las buenas trayectorias más probables

Por eso el algoritmo se llama **REINFORCE** — refuerza acciones proporcionalmente a qué tan bien funcionaron.

### En código PyTorch

```python
loss = -(log_probs * returns).sum()
loss.backward()       # autograd calcula ∇ por nosotros
optimizer.step()      # gradient ascent (el signo menos convierte descent en ascent)
```

---

## 9) Funciones de valor y la función de ventaja

### Cuatro funciones de valor (del artículo de Wolfe)

| Función | Nombre | Significado |
|---------|--------|-----------|
| `Vπ(s)` | Valor de estado | Return esperado comenzando desde estado `s`, siguiendo policy `π`. "¿Qué tan bueno es estar aquí?" |
| `Qπ(s,a)` | Valor de acción | Return esperado desde `s`, tomando acción `a`, luego siguiendo `π`. "¿Qué tan buena es esta acción específica aquí?" |
| `V*(s)` | Valor de estado óptimo | Igual pero asumiendo la mejor policy posible |
| `Q*(s,a)` | Valor de acción óptimo | Igual con policy óptima. **Esto es lo que DQN aprende.** |

### La función de ventaja

```
A(s,a)  =  Q(s,a) - V(s)
```

"¿Cuánto **mejor** es la acción `a` comparada con la acción **promedio** en el estado `s`?"

- `A > 0` → mejor que el promedio → reforzar esta acción
- `A < 0` → peor que el promedio → desincentivar esta acción

### Por qué importa la ventaja
REINFORCE usa el return crudo `G` como peso — ruidoso porque **todas** las acciones reciben crédito por el outcome total del episodio.
Usando ventaja: solo las acciones que fueron **MEJORES que lo esperado** se refuerzan positivamente.
Esto reduce dramáticamente la varianza del gradiente.

### El vanilla policy gradient (basado en ventaja)

```
∇J(θ) = E[ ∇log π(a|s) · A(s,a) ]
```

Misma estructura que REINFORCE, pero `G` se reemplaza con `A(s,a)`.
**Esto es lo que A2C y PPO realmente optimizan.**

### Conexión con RLHF
PPO — el algoritmo de RL detrás de ChatGPT — usa exactamente este policy gradient basado en ventaja con actualizaciones clipeadas y GAE (Generalized Advantage Estimation, Schulman 2015).

**Evolución**: `REINFORCE → + baseline V(s) → ventaja → A2C → PPO`

### Mapeo a código

```python
# benchmarks/a2c.py
advantage = returns - value_net(states)
policy_loss = -(log_probs * advantage)
```

---

## 10) Variantes del policy gradient — reduciendo varianza paso a paso

Presentar estas cuatro variantes como una progresión. Cada una mantiene el mismo valor esperado pero reduce ruido:

### Variante 1: Básica (REINFORCE)

```
∇J = E[ Σₜ ∇log π(a|s) · R(τ) ]
```

Peso = **return total de trayectoria** `R(τ)`.
Problema: rewards pasados (antes de que se tomara la acción) agregan ruido sin señal útil.

### Variante 2: Reward-to-Go

```
∇J = E[ Σₜ ∇log π(aₜ|sₜ) · Gₜ ]
```

Peso = **return desde timestep t en adelante solamente**.
Los rewards pasados se eliminan — una acción solo debería juzgarse por lo que pasa **después** de ella.
Mismo valor esperado, **menor varianza**.

> **Esto es lo que `benchmarks/policy_gradient.py` implementa vía `_discounted_returns()`.**

### Variante 3: Sustracción de baseline

```
∇J = E[ Σₜ ∇log π · (Gₜ - b(s)) ]
```

Restar un baseline `b(s)` que solo depende del estado.
Opción común: `b(s) = V(s)` (la función de valor de estado).
Solo las acciones con returns **sobre el promedio** se refuerzan positivamente.

### Variante 4: Vanilla Policy Gradient (Ventaja)

```
∇J = E[ Σₜ ∇log π · Aπ(s,a) ]
```

Usar ventaja `A = Q(s,a) - V(s)` como peso. **Menor varianza** de las cuatro.

> **Esto es lo que A2C (`benchmarks/a2c.py`) y PPO (`benchmarks/ppo.py`) implementan.**

### Insight matemático clave
Las cuatro variantes tienen el **MISMO valor esperado** (son estimadores insesgados del verdadero policy gradient).
La diferencia es **varianza**: cada paso elimina ruido sin cambiar lo que optimizamos.

```
Menor varianza → menos episodios necesarios → convergencia más rápida → entrenamiento más estable
```

### GAE (Generalized Advantage Estimation)
En la práctica, estimar `A(s,a)` exactamente es difícil. GAE (Schulman 2015) mezcla estimaciones TD multi-paso con un parámetro `λ` para balancear sesgo y varianza. PPO usa GAE por defecto.

> **Código**: `benchmarks/ppo.py` usa GAE en su cómputo de rollout buffer.

---

## 11) Monte Carlo en REINFORCE

### Qué significa Monte Carlo aquí
REINFORCE espera hasta el final del episodio, luego calcula returns completos de las recompensas muestreadas.
No se usa bootstrap ni target de valor en la versión base.

### Por qué es útil
- Estimación insesgada del return para la policy muestreada
- Implementación muy simple

### Por qué es difícil
- Updates de alta varianza
- El aprendizaje puede ser inestable y más lento

### Ejemplo numérico

Rewards del episodio: `[1, 1, 1]`, con `γ = 0.9`

```
G₀ = 1 + 0.9×1 + 0.9²×1  = 1 + 0.9 + 0.81  = 2.71
G₁ = 1 + 0.9×1            = 1 + 0.9          = 1.90
G₂ = 1                                        = 1.00
```

La acción más temprana recibe mayor peso porque influencia más recompensas futuras.

### Normalización de returns
Práctica común: normalizar returns por episodio.

```python
G_norm = (G - mean(G)) / std(G)
```

Esto asegura que aproximadamente la mitad de las acciones se incentivan y la otra mitad se desincentivan — estabilizando magnitudes de gradiente. Funciona como aproximación simple de baseline.

### Trick de reward-to-go
Solo las recompensas futuras importan para cada acción:
Usar `G` desde timestep `t` en adelante (no desde el inicio del episodio). Reduce varianza sin agregar sesgo.

---

## 12) Algoritmo REINFORCE (paso a paso)

```
1. Reset del environment y muestrear un episodio con la policy actual
2. En cada paso, guardar log π(aₜ|sₜ) y reward rₜ
3. Al terminar el episodio, calcular returns descontados:
       Gₜ = rₜ + γ · Gₜ₊₁     (caminar hacia atrás)
4. (Opcional) normalizar returns:
       G = (G - mean) / std
5. Calcular policy loss:
       L = -Σₜ log π(aₜ|sₜ) · Gₜ
6. Backpropagate y actualizar parámetros con Adam:
       loss.backward()
       optimizer.step()
7. Repetir por muchos episodios
```

**¿Por qué el signo menos?** PyTorch hace gradient **descent** por defecto. Nosotros queremos gradient **ascent** (maximizar return). El signo menos convierte descent en ascent.

---

## 13) Mapeo a código para Parte 1

### Implementación REINFORCE
- Config/dataclass: [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)
- Red de policy (`PolicyNetwork`): MLP de 2 capas (128 unidades ocultas, ReLU). Input: vector de estado. Output: logits de acción → softmax → distribución Categorical.
- Helper de returns (`_discounted_returns`): Recorre rewards hacia atrás calculando `G = r + γ*G_next`. Implementa la variante **reward-to-go**.
- Training loop (`run_policy_gradient`): Cada episodio: muestrear trayectoria → calcular returns → normalizar → calcular loss → backprop → step optimizer.
- `PolicyGradientConfig`: `gamma=0.99, lr=1e-3, episodes=700, hidden_size=128, normalize_returns=True`

### Correspondencia teoría-código

| Concepto teórico | Ubicación en código |
|-----------------|-------------------|
| `πθ(a\|s)` — policy estocástica | `PolicyNetwork.forward()` → `Categorical(logits)` |
| `log π(a\|s)` — log probabilidad | `dist.log_prob(action)` |
| `Gₜ` — return descontado (reward-to-go) | `_discounted_returns()` |
| Normalización (aprox. de baseline) | `(returns - mean) / std` en training loop |
| `L = -Σ log π · G` — policy loss | `-(log_probs * returns).sum()` |
| `∇θ` via autograd | `loss.backward(); optimizer.step()` |

---

## 14) Comandos de demo (Parte 1)

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

## 15) Notas didácticas para audiencia low-level

Secuencia sugerida en slides/pizarra:

1. Definir policy como probabilidades sobre acciones.
2. Explicar una trayectoria muestreada como "evidencia".
3. Mostrar return descontado y por qué acciones tempranas afectan más el futuro.
4. Mostrar policy gradient como "subir probabilidad de acciones con mejor outcome".
5. Introducir funciones de valor `V(s)` y `Q(s,a)` como "lo que esperamos en promedio".
6. Mostrar ventaja `A = Q - V` como "¿fue esta acción mejor o peor que el promedio?"
7. Recorrer las cuatro variantes: REINFORCE → reward-to-go → baseline → ventaja.
8. Aclarar trade-off Monte Carlo: simple + insesgado, pero ruidoso.
9. Conectar con RLHF: "PPO usa la variante basada en ventaja + clipping".

**Confusiones comunes a resolver explícitamente:**
- "¿Esto es supervised learning?" → No; no hay labels, hay rewards diferidas del entorno.
- "¿Por qué no elegir siempre argmax?" → La estocasticidad ayuda exploración y aprendizaje por gradiente.
- "¿Por qué normalizar returns?" → Para estabilizar magnitud de updates. Funciona como baseline simple.
- "¿Cuál es la diferencia entre G y A?" → G es return crudo, A es return menos baseline (cuánto mejor que lo esperado).

---

## 16) Puente a Parte 2

Parte 2 responde la siguiente pregunta natural tras REINFORCE:
- ¿Cómo reducir varianza y mejorar estabilidad/rendimiento?

Debilidades REINFORCE → Soluciones Parte 2:

| # | Debilidad REINFORCE | Solución Parte 2 |
|---|-------------------|----------------|
| 1 | Gradientes de alta varianza | **A2C**: agregar critic V(s) para calcular ventaja |
| 2 | Ineficiencia de muestras | **A3C**: workers paralelos para decorrelación de datos |
| 3 | Sin asignación de crédito | **PPO**: surrogate clipeado + GAE |
| 4 | Sensible a hiperparámetros | **TRPO**: trust region con restricción KL |
| 5 | Sin restricción en tamaño de update | **PPO**: clipea ratio para prevenir updates catastróficos |

Todos los métodos de Parte 2 se construyen directamente sobre la base de policy gradient cubierta hoy.