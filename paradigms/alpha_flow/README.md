# AlphaFlow

> Zhang et al., "AlphaFlow: Understanding and Improving MeanFlow Models", ICLR 2026.

AlphaFlow unifies flow matching and MeanFlow via a tunable parameter $\alpha$ that blends between two ways of computing the mean velocity target: a **continuous** target (JVP-based, as in MeanFlow) and a **discrete** target (Euler stepping-based). Setting $\alpha = 0$ recovers MeanFlow; $\alpha = 1$ uses a purely discrete target.

**Intuition**: MeanFlow computes the training target for the average velocity $u$ using a JVP — a continuous, infinitesimal derivative $du/dt$. This is mathematically elegant but can be noisy in practice. AlphaFlow introduces an alternative: a discrete target computed by taking one small Euler step of size $dt$ and composing the instantaneous velocity $v$ over $dt$ with the model's own prediction $u$ over the remaining interval $h - dt$. Blending the two targets via $\alpha$ gives a knob to trade off between the continuous and discrete approaches. The discrete target acts as a form of self-consistency (similar to shortcut models) and can stabilize training.

---

## Model Architecture

AlphaFlow reuses the same **MeanFlowMLPDenoiser** — a single-head MLP conditioned on $t$ and $h = t - r$, outputting the average velocity $u$.

```
MeanFlowMLPDenoiser:  (z_t ∈ R², t ∈ R, h ∈ R)  →  u ∈ R²

Conditioning:  c = t_embed(t) + h_embed(h)
Body:          z_t → Linear(2→256) → [AdaLN ResBlock × 6] → AdaLN → Linear(256→2) → u
```

---

## Two Target Formulations

For samples in the MeanFlow regime ($h = t - r > 0$), the target blends two formulations:

### Continuous Target (MeanFlow, $\alpha = 0$)

Uses JVP to compute $du/dt$ and constructs:

$$u_{\text{cont}} = v - h \cdot \text{sg}(du/dt)$$

This is the standard MeanFlow target — derived from the identity $u = v - h \cdot du/dt$.

```python
# alpha_flow.py: training_loss() — continuous target
u, du_dt = torch.func.jvp(u_fn, (z_t, t, r), (v, ones, zeros))
u_tgt = v - h * du_dt.detach()
```

### Discrete Target ($\alpha = 1$)

Takes one Euler step of size $dt$ using the true velocity $v$, then queries the model for the average velocity over the remaining interval:

$$z_{t-dt} = z_t - dt \cdot v$$

$$u_{\text{disc}} = \frac{dt \cdot v + (h - dt) \cdot \text{sg}(u(z_{t-dt}, t-dt, h-dt))}{h}$$

This is a weighted average: the true velocity over $dt$ and the model's prediction over $h - dt$.

```python
# alpha_flow.py: training_loss() — discrete target
with torch.no_grad():
    z_stepped = z_t - dt * v                          # one Euler step
    u_next = model(z_stepped, t - dt, h - dt)         # model on remaining interval

u_tgt_disc = (dt * v + (h - dt) * u_next) / h         # weighted average
```

### Blended Target

The final target blends the two, only for MeanFlow samples ($h > 0$):

$$u_{\text{target}} = (1 - \alpha) \cdot u_{\text{cont}} + \alpha \cdot u_{\text{disc}}$$

```python
# alpha_flow.py: training_loss()
u_tgt = u_tgt + alpha * (u_tgt_disc - u_tgt)          # blend (MF samples only)
loss = (u - u_tgt.detach()).pow(2).sum(dim=-1)
```

For flow matching samples ($h = 0$), both targets reduce to $v = \epsilon - x_0$.

---

## Training

The overall training structure is the same as MeanFlow:

1. **Sample** $t, r$ from logit-normal with $t \geq r$. 75% of the batch uses $r = t$ (FM regime).
2. **Construct** $z_t = (1-t) x_0 + t \epsilon$ and $v = \epsilon - x_0$.
3. **Compute** the blended target (JVP for continuous + Euler step for discrete).
4. **Loss**: adaptive-weighted MSE between $u$ and the blended target.

When $\alpha = 0$, no discrete step is computed and training is identical to MeanFlow.

---

## Sampling

Sampling is identical to MeanFlow — Euler integration using the average velocity:

$$x_0 = z - u(z, t=1, h=1) \quad \text{(one-step)}$$

```python
# alpha_flow.py: sample()
z = torch.randn(shape, device=device)
t_steps = linspace(1.0, 0.0, num_steps + 1)

for i in range(num_steps):
    t, r = t_steps[i], t_steps[i + 1]
    h = t - r
    u = model(z, t, h)
    z = z - h * u
```

---

## Connection to MeanFlow and Shortcut Models

AlphaFlow sits between MeanFlow and shortcut models. At $\alpha = 0$, it is exactly MeanFlow (continuous JVP target). As $\alpha$ increases toward 1, the discrete target introduces a form of self-consistency similar to shortcut models: the model's own prediction at a nearby point is used to construct the target. The key difference is that shortcut models operate in the flow matching velocity space and condition on step count, while AlphaFlow operates in the mean velocity space and blends continuous and discrete targets.
