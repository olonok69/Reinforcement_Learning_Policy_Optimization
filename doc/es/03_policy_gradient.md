# Policy Gradient (REINFORCE)

## Idea central
En lugar de aprender $Q(s,a)$, policy gradients optimiza directamente una policy estocástica parametrizada $\pi_\theta(a|s)$ maximizando expected return:

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

Update REINFORCE (Monte Carlo):

$$
\nabla_\theta J(\theta) \approx \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\,G_t
$$

Interpretación:
- $\log \pi_\theta(a_t|s_t)$ aumenta la probabilidad de las acciones muestreadas.
- $G_t$ actúa como señal de calidad de esas acciones.
- Trayectorias buenas aumentan su propia probabilidad; trayectorias malas la reducen.

Como $G_t$ se estima con returns de episodio completo, la varianza suele ser alta, especialmente al inicio.

## Cómo lo implementa este repo
- Módulo: `benchmarks/policy_gradient.py`
- Runner: `policy_gradient_benchmark.py`
- Environment: `CartPole-v1`
- Usa returns por episodio y normalización opcional de returns.

## Flujo de entrenamiento
1. Hacer rollout completo de un episodio con acciones estocásticas.
2. Calcular discounted returns por timestep.
3. Ponderar log-probabilities con returns.
4. Backpropagate de policy loss.

En la práctica, este método es un estimador no sesgado pero ruidoso del gradiente de policy.

## Por qué funciona (resumen corto)
El truco de la derivada logarítmica transforma gradientes sobre probabilidades de trayectorias en una esperanza de términos score-function, permitiendo optimizar expected return con rollouts muestreados. REINFORCE es la forma más directa de esta idea y por eso suele ser el punto de partida conceptual para actor-critic y PPO.

## Fortalezas
- Maneja policies estocásticas de forma natural.
- Funciona bien en espacios de policy continuos y de alta dimensión.
- Puente conceptual simple hacia actor-critic.

## Limitaciones
- Gradientes de alta varianza.
- Baja sample efficiency (Monte Carlo por episodio completo).
- Sensible a baseline/normalización.

## Palancas de ajuste comunes
- **Learning rate**: muy alta produce oscilación inestable; muy baja ralentiza aprendizaje.
- **Discount factor $\gamma$**: valores altos priorizan horizonte largo pero elevan varianza.
- **Normalización de returns**: mejora consistencia de escala en optimización.
- **Coeficiente de entropía** (si se usa): aumenta exploración, pero demasiado puede frenar convergencia.

## Notas prácticas
- Normalizar returns suele mejorar estabilidad.
- Entropy regularization (no obligatoria) puede mejorar exploración.
- A2C/PPO suelen ser sucesores más estables en la práctica.
