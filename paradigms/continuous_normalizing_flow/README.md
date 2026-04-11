# Continuous Normalizing Flow (FFJORD)

> Grathwohl et al., "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models", ICLR 2019.

A CNF replaces the discrete stack of invertible layers (as in RealNVP) with a **continuous-time ODE**. A neural network $v_\theta(x, t)$ defines a velocity field, and the transformation from data to latent space is the solution of this ODE. The log-density change is computed via the **instantaneous change of variables** formula.

**Intuition**: A discrete normalizing flow warps space through a fixed sequence of carefully designed invertible layers. A CNF takes this idea to the limit — instead of discrete layers, the transformation is a smooth, continuous flow governed by an ODE. Think of a velocity field in a fluid: data points are like particles carried by the current, smoothly flowing from the data distribution at $t=0$ to a Gaussian at $t=1$. A neural network learns this velocity field, and integration traces each particle's path. The key advantage over discrete flows is architectural freedom: the velocity network can be *any* neural network (no invertibility constraints, no coupling layer tricks), because the ODE is inherently invertible — just integrate backward. The cost is that training requires backpropagating through the ODE integration steps and computing the Jacobian trace at each step.

---

## Model Architecture

The velocity field $v_\theta(x, t)$ is parameterized by the same MLPDenoiser used for diffusion and flow matching — it takes a 2D position and a time scalar, and outputs a 2D velocity.

```
Velocity network:  (x ∈ R², t ∈ R)  →  MLPDenoiser  →  v ∈ R²

Training (data → latent):   x₀ ──[Euler: x += v·dt, accumulate Tr(∂v/∂x)·dt]──→ x_T ≈ N(0,I)
                            t: 0 → 1

Sampling (latent → data):   z ~ N(0,I) ──[Euler: x -= v·dt]──→ x₀
                            t: 1 → 0
```

```python
# cnf.py: ContinuousNormalizingFlow
# Training: Euler integration forward, accumulating log-det via exact trace
x = x0.detach().requires_grad_(True)
log_det = zeros(B)
for i in range(num_integration_steps):               # default: 10 steps
    v = model(x, t_batch)                            # v_θ(x_t, t)
    trace = _exact_trace_2d(v, x)                    # Tr(∂v/∂x) via autograd
    log_det += trace * dt
    x = x + v * dt                                   # Euler step

# Sampling: Euler integration backward (no trace needed)
x = randn(shape)                                     # z ~ N(0, I)
for i in range(num_steps):
    v = model(x, t_batch)                            # t: 1 → 0
    x = x - v * dt                                   # Reverse Euler step
```

---

## Continuous Change of Variables

For a continuous transformation $\frac{dx}{dt} = v_\theta(x, t)$, the log-density evolves as:

$$\frac{d \log p(x_t)}{dt} = -\text{Tr}\!\left(\frac{\partial v_\theta}{\partial x}\right)$$

Integrating from $t=0$ (data) to $t=1$ (latent):

$$\log p(x_0) = \log p(x_1) + \int_0^1 \text{Tr}\!\left(\frac{\partial v_\theta(x_t, t)}{\partial x}\right) dt$$

This replaces the discrete log-determinant sum in normalizing flows with a continuous integral of the Jacobian trace.

---

## Exact Trace Computation (2D)

For 2D data, the trace of the Jacobian $\frac{\partial v}{\partial x}$ is computed exactly using autograd:

$$\text{Tr}\!\left(\frac{\partial v}{\partial x}\right) = \frac{\partial v_1}{\partial x_1} + \frac{\partial v_2}{\partial x_2}$$

This requires only 2 autograd calls (one per dimension):

```python
# cnf.py: _exact_trace_2d()
trace = torch.zeros(x.shape[0], device=x.device)
for d in range(dim):
    grad_d = torch.autograd.grad(
        v[:, d].sum(), x, create_graph=True, retain_graph=True
    )[0]
    trace = trace + grad_d[:, d]    # Diagonal element: ∂v_d/∂x_d
```

> For higher dimensions, the Hutchinson trace estimator $\text{Tr}(A) \approx \epsilon^T A \epsilon$ would be used instead of exact computation.

---

## Training (Negative Log-Likelihood)

Training minimizes the NLL by integrating the ODE forward (data → latent) using Euler steps, accumulating the trace:

$$\text{NLL} = -\log p_z(x_T) - \int_0^1 \text{Tr}\!\left(\frac{\partial v_\theta}{\partial x}\right) dt$$

```python
# cnf.py: training_loss()
x = x0.detach().requires_grad_(True)         # Enable grad for Jacobian computation
log_det = torch.zeros(batch_size, device=device)
dt = 1.0 / self.num_integration_steps

for i in range(self.num_integration_steps):
    t_val = i * dt
    t_batch = torch.full((batch_size,), t_val, device=device)

    v = model(x, t_batch, **kwargs)           # v_θ(x_t, t)
    trace = _exact_trace_2d(v, x)             # Tr(∂v/∂x)

    log_det = log_det + trace * dt            # Accumulate ∫ Tr dt
    x = x + v * dt                            # Euler step: x_{t+dt} = x_t + v·dt
```

After integration, $x \approx x_T$ (should be close to standard normal):

```python
# cnf.py: training_loss() — final NLL
log_prior = -0.5 * (x ** 2 + math.log(2 * math.pi)).sum(dim=-1)  # log N(x_T; 0, I)
nll = -log_prior - log_det
return nll.mean()
```

### Gradient Flow

A key detail: `x.detach().requires_grad_(True)` creates a fresh leaf tensor so that `torch.autograd.grad` can compute the Jacobian trace. The computation graph then flows through the Euler steps, allowing gradients to reach the model parameters.

---

## Sampling

Generate samples by solving the ODE **backward** from $t=1$ (noise) to $t=0$ (data):

$$x_{t-\Delta t} = x_t - \Delta t \cdot v_\theta(x_t, t)$$

```python
# cnf.py: sample()
x = torch.randn(shape, device=device)           # x_T ~ N(0, I)
dt = 1.0 / num_steps

for i in range(num_steps):
    t_val = 1.0 - i * dt                        # t: 1 → 0
    t_batch = torch.full((batch_size,), t_val, device=device)
    v = model(x, t_batch, **kwargs)
    x = x - v * dt                               # Reverse Euler step
```

Note: sampling does **not** need the trace computation — only the velocity field.

---

## Connection to Discrete Normalizing Flows

A CNF is the continuous-time generalization of a discrete normalizing flow. Where RealNVP stacks a finite number of invertible coupling layers and sums their log-determinants, a CNF replaces this with an ODE whose solution defines a single smooth transformation — and the discrete log-det sum becomes a continuous integral of the Jacobian trace. This removes the need for specially designed invertible architectures (coupling layers, autoregressive masks), since any neural network can parameterize the velocity field. The tradeoff is computational: training requires backpropagating through ODE integration steps and computing the Jacobian trace at each step, whereas discrete flows only need a single forward pass per layer.

## Connection to Flow Matching

CNF and flow matching both learn a velocity field $v_\theta(x, t)$ and sample by solving the same ODE backward. The difference is entirely in how they train. A CNF maximizes exact log-likelihood by integrating the full ODE forward and accumulating the Jacobian trace — this is principled but expensive (sequential Euler steps, autograd per step). Flow matching sidesteps this entirely: it regresses $v_\theta$ directly against a target velocity using a simple MSE loss on individual $(x, t)$ pairs, with no ODE integration or Jacobian computation during training. This makes flow matching much cheaper to train, but it gives up exact likelihood computation — you get a generative model that samples well but can't evaluate $\log p(x)$.
