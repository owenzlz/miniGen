# MeanFlow

> Geng et al., "Mean Flows for One-step Generative Modeling", arXiv:2505.13447.

MeanFlow learns an **average velocity field** $u(z_t, t, h)$ that averages the instantaneous flow matching velocity over a time interval $[r, t]$, where $h = t - r$. This enables **exact one-step generation**: a single evaluation of $u$ at $h = 1$ gives the mean velocity over the entire trajectory, jumping from noise to data in one step.

**Intuition**: In standard flow matching, the model learns the instantaneous velocity $v$ at each point in time — you need many small Euler steps to follow the trajectory. MeanFlow instead learns the *average* velocity over a time window. If you know the average velocity from $t=1$ to $t=0$, you can take one big step: $x_0 = z - u(z, t=1, h=1)$. The challenge is that the average velocity depends on the interval, so the model must be conditioned on both $t$ and $h = t - r$. Training uses JVP (Jacobian-vector product) to compute $du/dt$ and constructs a self-consistent target.

---

## Model Architecture

MeanFlowMLPDenoiser is a single-head MLP conditioned on both $t$ (noise level) and $h = t - r$ (interval width), outputting the average velocity $u$.

```
MeanFlowMLPDenoiser:  (z_t ∈ R², t ∈ R, h ∈ R)  →  u ∈ R²

Conditioning:  c = t_embed(t) + h_embed(h)
Body:          z_t → Linear(2→256) → [AdaLN ResBlock × 6] → AdaLN → Linear(256→2) → u
```

```python
# meanflow_mlp.py: MeanFlowMLPDenoiser
def forward(self, x, t, h):
    feat = self.input_proj(x)
    c = self.t_embed(t) + self.h_embed(h)         # condition on both t and h
    for block in self.blocks:
        feat = block(feat, c)                      # AdaLN residual blocks
    return self.output_proj(feat)                   # single output: u
```

---

## The Average Velocity

The average velocity over $[r, t]$ is defined as:

$$u(z_t, r, t) = \frac{1}{t - r} \int_r^t v(z_s, s) ds$$

where $v$ is the instantaneous velocity. Taking the derivative with respect to $t$:

$$\frac{du}{dt} = \frac{v - u}{t - r}$$

Rearranging gives the key identity used in training:

$$u = v - (t - r) \frac{du}{dt}$$

---

## Training Loss

The loss makes the model's output $u$ self-consistent with the instantaneous velocity $v = \epsilon - x_0$:

1. **Sample time pairs**: $t, r$ from logit-normal, with $t \geq r$. A fraction of the batch (75%) uses $r = t$ (pure flow matching regime).

2. **Construct noisy sample**: $z_t = (1-t) x_0 + t \epsilon$

3. **Compute $u$ and $du/dt$ via JVP**: The Jacobian-vector product efficiently computes $du/dt$ in one forward pass by differentiating $u$ with respect to $t$ along the direction of $v$:

```python
# meanflow.py: training_loss()
def u_fn(z_t, t, r):
    return model(z_t, t, h=t-r)

u, du_dt = torch.func.jvp(u_fn, (z_t, t, r), (v, ones, zeros))
```

4. **Construct target and compute loss**:

$$u_{\text{target}} = v - (t - r) \cdot \text{sg}(du/dt)$$

$$\mathcal{L} = \|u - u_{\text{target}}\|^2$$

The stop-gradient on $du/dt$ prevents the model from trivially satisfying the constraint by collapsing.

```python
# meanflow.py: training_loss()
u_tgt = v - (t - r) * du_dt.detach()               # stop-gradient on du/dt
loss = (u - u_tgt).pow(2).sum(dim=-1)               # per-sample MSE
loss = adaptive_weight(loss).mean()                  # adaptive weighting
```

---

## Sampling

### One-Step (num_steps=1)

The average velocity over $[0, 1]$ gives an exact one-step mapping:

$$x_0 = z - u(z, t=1, h=1), \quad z \sim \mathcal{N}(0, I)$$

### Multi-Step Euler

For higher quality, use multiple steps with the average velocity over each sub-interval:

```python
# meanflow.py: sample()
z = torch.randn(shape, device=device)
t_steps = linspace(1.0, 0.0, num_steps + 1)

for i in range(num_steps):
    t, r = t_steps[i], t_steps[i + 1]
    h = t - r
    u = model(z, t, h)
    z = z - h * u                                    # step by h * u (not dt * v)
```

Note: each step uses $h = t - r$ as both the interval width for the model and the step size — the average velocity $u$ is already scaled for this interval.

---

## Connection to Flow Matching

MeanFlow and flow matching learn different quantities from the same interpolant. Flow matching learns the instantaneous velocity $v$ and needs many Euler steps at inference. MeanFlow learns the average velocity $u$ over a window, enabling one-step generation by construction. When $r = t$ (i.e., $h = 0$), the average velocity reduces to the instantaneous velocity, so a fraction of the training batch operates in the standard flow matching regime.
