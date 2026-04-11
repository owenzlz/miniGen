# Flow Matching

> Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
> Liu et al., "Flow Straight and Fast", ICLR 2023.

(Rectified) Flow Matching learns a **velocity field** $v_\theta(x, t)$ that transports samples along a probability path from noise ($t=1$) to data ($t=0$). Unlike diffusion models, it uses continuous time and ODE-based sampling (no stochastic noise at inference).

**Intuition**: Imagine you have a cloud of random noise points and a cloud of data points, and you want to morph one into the other. The simplest plan: draw a straight line from each noise point to its paired data point, and slide along it at constant speed. The velocity at any point along any line is just "destination minus origin." Flow matching trains a neural network to predict this velocity everywhere in space and time. At inference, you drop noise points and integrate the learned velocity field — each point flows along its predicted trajectory to land in data space. Because the training target is a simple MSE on velocities at random $(x_t, t)$ pairs, there's no ODE integration or Jacobian computation during training — just a single forward pass — making it far cheaper to train than a CNF while sampling the same way.

---

## Model Architecture

The velocity network $v_\theta(x, t)$ uses the same MLPDenoiser as diffusion and CNF — it takes a 2D position and a time scalar, and outputs a 2D velocity.

```
Velocity network:  (x_t ∈ R², t ∈ R)  →  MLPDenoiser  →  v ∈ R²

Training:   x₀ ~ p_data, x₁ ~ N(0,I), t ~ U[0,1]
            x_t = (1-t)·x₀ + t·x₁           ← linear interpolant
            target v_t = x₁ - x₀             ← constant velocity
            loss = ||v_θ(x_t, t) - v_t||²    ← simple MSE, no ODE needed

Sampling:   z ~ N(0,I) ──[ODE: dx/dt = v_θ(x,t)]──→ x₀
            t: 1 → 0  (Euler or Heun solver)
```

```python
# flow_matching.py: FlowMatching
def training_loss(self, model, x0, t):
    xt, velocity_target, _ = self.forward_process(x0, t)   # interpolate + target
    velocity_pred = model(xt, t)                            # v_θ(x_t, t)
    return F.mse_loss(velocity_pred, velocity_target)       # ||v_θ - v_t||²

def sample(self, model, shape, device, num_steps=50):
    x = torch.randn(shape, device=device)                   # z ~ N(0, I)
    x = self.solver.solve(model, x, t_start=1.0, t_end=0.0, num_steps=num_steps)
    return x
```

---

## Probability Path (Interpolant)

An **interpolant** $I(x_0, x_1, t)$ defines how data and noise are mixed at time $t$:

$$x_t = I(x_0, x_1, t), \quad x_0 \sim p_{\text{data}}, \quad x_1 \sim \mathcal{N}(0, \mathbf{I}), \quad t \in [0, 1]$$

The target velocity is its time derivative:

$$v_t = \frac{dI}{dt}(x_0, x_1, t)$$

### Linear Interpolant (Optimal Transport Path)

The simplest and most common choice — a straight line between data and noise:

$$x_t = (1 - t) x_0 + t x_1$$

$$v_t = \frac{dx_t}{dt} = x_1 - x_0$$

The velocity is **constant** (independent of $t$), which encourages straight trajectories.

```python
# interpolant/linear.py
def __call__(self, x0, x1, t):
    t = self._reshape_t(t, x0)
    return (1 - t) * x0 + t * x1    # x_t = (1-t)x_0 + t*x_1

def velocity(self, x0, x1, t):
    return x1 - x0                   # v_t = x_1 - x_0
```

---

## Training

The model $v_\theta(x_t, t)$ is trained to match the target velocity via MSE:

$$\mathcal{L} = \mathbb{E}_{t \sim U[0,1], x_0, x_1}\bigl[\|v_\theta(x_t, t) - v_t\|^2\bigr]$$

Steps:
1. Sample $x_0 \sim p_{\text{data}}$, $x_1 \sim \mathcal{N}(0, I)$, $t \sim U[0,1]$
2. Compute $x_t$ and target $v_t$ from the interpolant
3. Minimize $\|v_\theta(x_t, t) - v_t\|^2$

```python
# flow_matching.py: training_loss()
xt, velocity_target, _ = self.forward_process(x0, t)  # Step 1-2
velocity_pred = model(xt, t, **kwargs)                  # Model prediction
loss = F.mse_loss(velocity_pred, velocity_target)       # Step 3
```

```python
# flow_matching.py: forward_process()
noise = torch.randn_like(x0)                    # x_1 ~ N(0, I)
xt = self.interpolant(x0, noise, t)              # x_t = I(x_0, x_1, t)
velocity = self.interpolant.velocity(x0, noise, t)  # v_t = dI/dt
```

Timesteps are sampled **continuously** from $[0, 1]$ (unlike discrete DDPM):

```python
# flow_matching.py: sample_timesteps()
return torch.rand(batch_size, device=device)     # t ~ U[0, 1]
```

---

## Sampling (ODE Solving)

Generate samples by solving the ODE **backward** from noise to data:

$$\frac{dx}{dt} = v_\theta(x, t), \quad x(1) \sim \mathcal{N}(0, \mathbf{I}) \longrightarrow x(0) \approx p_{\text{data}}$$

```python
# flow_matching.py: sample()
x = torch.randn(shape, device=device)               # x_1 ~ N(0, I)
x = self.solver.solve(model, x, t_start=1.0, t_end=0.0, num_steps=num_steps)
```

### Euler Solver (1st order)

The simplest integrator. One model evaluation per step:

$$x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)$$

For sampling from $t=1 \to 0$, $\Delta t < 0$.

```python
# solver/euler.py: step()
v = model(x, t, **kwargs)       # v_θ(x_t, t)
return x + dt * v               # x_{t+dt} = x_t + dt * v
```

### Heun Solver (2nd order)

Predictor-corrector method. Two model evaluations per step, but more accurate:

$$\tilde x = x_t + \Delta t \cdot v_\theta(x_t, t) \quad \text{(Euler predictor)}$$

$$x_{t+\Delta t} = x_t + \frac{\Delta t}{2}\bigl(v_\theta(x_t, t) + v_\theta(\tilde x, t + \Delta t)\bigr) \quad \text{(trapezoidal corrector)}$$

```python
# solver/heun.py: step()
k1 = model(x, t, **kwargs)              # v_θ(x_t, t)
x_pred = x + dt * k1                    # Euler predictor
k2 = model(x_pred, t + dt, **kwargs)    # v_θ(x̃, t+dt)
return x + dt * 0.5 * (k1 + k2)         # Trapezoidal corrector
```

### Solver Loop

Both solvers share the same integration loop:

```python
# solver/base.py: solve()
dt = (t_end - t_start) / num_steps      # Negative for t: 1→0
t = t_start
for _ in range(num_steps):
    x = self.step(model, x, t_batch, dt_batch, **kwargs)
    t = t + dt
```

---

## Connection to Diffusion

Flow matching and DDPM both learn to reverse a noise-corruption process, but they differ in almost every design choice. DDPM uses discrete timesteps with a fixed noise schedule $\beta_t$ and trains a network to predict the added noise $\epsilon$; sampling is stochastic (ancestral sampling adds fresh noise at each step). Flow matching uses continuous time $t \in [0, 1]$ with a pluggable interpolant defining the corruption path, and trains a network to predict the velocity $v_t$; sampling is deterministic ODE integration. The result is that flow matching is more modular — you can swap interpolants and solvers independently — and the straight-line paths of the linear interpolant tend to need fewer integration steps than the curved paths induced by DDPM's noise schedule.

## Connection to CNF

Flow matching and CNF both learn a velocity field $v_\theta(x, t)$ and sample by solving the same ODE backward from noise to data. The difference is purely in training. A CNF maximizes exact log-likelihood by integrating the full ODE forward and accumulating the Jacobian trace at every step — principled but expensive. Flow matching replaces this with a simple MSE regression: sample a random $(x_t, t)$ pair, compute the target velocity from the interpolant, and minimize $\|v_\theta(x_t, t) - v_t\|^2$. No ODE integration, no Jacobian — just one forward pass per training step. The cost is that flow matching cannot evaluate $\log p(x)$, but in practice the velocity fields it learns are equally good for generation.
