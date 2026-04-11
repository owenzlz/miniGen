# Shortcut Model

> Frans et al., "One Step Diffusion via Shortcut Models", 2024.

A Shortcut Model extends flow matching by conditioning the velocity network on **step size**, enabling a single model to generate samples in anywhere from 1 to $N$ steps. Training combines standard flow matching loss with a **self-consistency** loss that teaches the model to take large steps by composing smaller ones.

![Shortcut Model](./shortcut_model_diagram.png)

**Intuition**: A standard flow matching model trained at many steps can't just use fewer steps at inference — it never learned what a "big step" looks like. A shortcut model fixes this by conditioning on step size and adding a self-consistency loss: a single large step from A to B must match two half-steps A→C→B. The result is one network that generates in 1 step (fast) or many steps (high-quality), with a smooth tradeoff in between.

---

## Model Architecture

The ShortcutMLPDenoiser extends the base MLPDenoiser with an additional **step-size embedding**. The conditioning vector becomes the sum of time, step-size, and (optionally) class embeddings.

```
ShortcutMLPDenoiser:  (x ∈ R², t ∈ R, d ∈ R)  →  v ∈ R²

Conditioning:  c = time_embed(t) + step_embed(d) [+ class_embed(y)]
               ↓
Body:          x → Linear(2→256) → [AdaLN ResBlock × 6] → AdaLN → Linear(256→2) → v
```

The step-size embedding has the same structure as the time embedding (sinusoidal → MLP), but is **zero-initialized** so that at the start of training, `d=0` contributes nothing — the model begins as a standard flow matching network.

```python
# shortcut_mlp.py: ShortcutMLPDenoiser
self.step_embed = nn.Sequential(
    SinusoidalEmbedding(hidden_size),
    nn.Linear(hidden_size, hidden_size), GELU(),
    nn.Linear(hidden_size, hidden_size),           # zero-initialized
)

def forward(self, x, t, d=None, y=None):
    c = self.time_embed(t) + self.step_embed(d)    # c = time + step-size
    # ... residual blocks with AdaLN conditioning on c ...
```

---

## The dt_base Representation

Step size is represented as $d = \log_2(N)$, not the raw step size $\Delta t = 1/N$:

$$d = \log_2(N)$$

where $N$ is the number of inference steps. For example: $d=0$ means one-step generation ($\Delta t = 1$), $d=1$ means two steps ($\Delta t = 1/2$), and $d=7$ means 128 steps ($\Delta t = 1/128$, the base FM resolution). This log-scale representation spaces out the step sizes evenly for the embedding network — each increment of $d$ halves the step size.

---

## Training

Training combines two losses in a single batch. With `bootstrap_every=4`, 25% of each batch trains self-consistency and 75% trains standard flow matching.

### Base Flow Matching Loss (75% of batch)

Standard velocity regression at the finest resolution ($d = \log_2 N$):

$$\mathcal{L}_{\text{FM}} = \|v_\theta(x_t, t, d=\log_2 N) - (x_1 - x_0)\|^2$$

```python
# shortcut_model.py: training_loss() — flow matching portion
x_t_flow = (1 - t) * x0 + t * z                    # linear interpolant
v_target_flow = z - x0                              # standard FM velocity
dt_base_flow = log2(N)                              # finest resolution
```

### Self-Consistency Loss (25% of batch)

A big step should match the composition of two half-steps. For a step of size $\Delta t$ at resolution $d$:

1. Take two half-steps at resolution $d+1$ (stop-gradient):

$$v_1 = v_\theta(x_t, t, d+1), \quad x_{\text{mid}} = x_t - \frac{\Delta t}{2} v_1$$

$$v_2 = v_\theta(x_{\text{mid}}, t - \frac{\Delta t}{2}, d+1)$$

2. The target for the full step is the average:

$$v_{\text{target}} = \text{sg}\left(\frac{v_1 + v_2}{2}\right)$$

3. Train the model to match this with a single big step at resolution $d$:

$$\mathcal{L}_{\text{SC}} = \|v_\theta(x_t, t, d) - v_{\text{target}}\|^2$$

```python
# shortcut_model.py: training_loss() — self-consistency portion
dt_base_bst = randint(0, log2_N)                    # random resolution
dt_base_target = dt_base_bst + 1                    # half-step resolution

with torch.no_grad():                               # stop-gradient targets
    v1 = model(x_t, t, d=dt_base_target)
    x_mid = (x_t - dt_half * v1).clamp(-4, 4)       # intermediate point
    v2 = model(x_mid, t - dt_half, d=dt_base_target)
    v_target = ((v1 + v2) / 2).clamp(-4, 4)         # averaged velocity
```

### Combined Forward Pass

Both portions are concatenated and processed in a **single forward pass** — efficient because the two stop-gradient calls plus one gradient call is cheaper than separate passes:

```python
# shortcut_model.py: training_loss() — merge and compute loss
x_t = cat([x_t_bst, x_t_flow])
v_target = cat([v_target_bst, v_target_flow])
dt_all = cat([dt_base_bst, dt_base_flow])

v_pred = model(x_t, t_all, d=dt_all)                # single forward pass
loss = F.mse_loss(v_pred, v_target)
```

---

## Sampling

Sampling is identical to flow matching ODE integration, but with the step-size $d$ passed to the model. The same network works at any step budget:

```python
# shortcut_model.py: sample()
x = torch.randn(shape, device=device)                # z ~ N(0, I)
dt_base = math.log2(num_steps)                        # e.g., 0 for one-step
delta_t = 1.0 / num_steps

for i in range(num_steps):
    t_val = 1.0 - i * delta_t                        # t: 1 → 0
    v = model(x, t_batch, d=d_batch)
    x = x - delta_t * v                              # Euler step
```

With `num_steps=1` ($d=0$), the model takes a single large step from noise to data. With `num_steps=128` ($d=7$), it behaves like standard flow matching with 128 Euler steps.

---

## Connection to Flow Matching

A shortcut model is a strict superset of flow matching. When $d = \log_2 N$ (the finest resolution), the self-consistency loss is inactive and training reduces to standard flow matching velocity regression. The shortcut model adds the ability to generate at coarser resolutions by learning how to compose fine steps into larger ones via self-consistency. At inference, you choose the step budget — 1 step for speed, many steps for quality — without retraining. The zero-initialized step embedding ensures the model starts as a standard flow matching network and gradually learns the shortcut behavior.
