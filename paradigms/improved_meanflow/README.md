# Improved MeanFlow (iMF)

> Geng et al., "Improved Mean Flows: On the Challenges of Fastforward Generative Models", arXiv:2512.02012.

Improved MeanFlow reformulates the MeanFlow loss so the regression target is **network-independent** ($v_{\text{target}} = \epsilon - x_0$), making training more stable. It uses a **dual-head architecture** with separate heads for the average velocity $u$ and instantaneous velocity $v$, and constructs a compound function $V = u + (t-r) \cdot \text{sg}(du/dt)$ that is regressed toward the target.

**Intuition**: The original MeanFlow's target $u_{\text{tgt}} = v - (t-r) \cdot \text{sg}(du/dt)$ depends on the current model's output through $du/dt$, creating a moving target that can be unstable. iMF flips the equation: instead of making $u$ match a model-dependent target, it constructs a compound function $V$ from $u$ and $du/dt$, and regresses $V$ toward the fixed target $\epsilon - x_0$. This is more stable because the target never changes. The dual-head architecture adds a v-head that learns the instantaneous velocity independently, providing a better signal for the JVP computation.

---

## Model Architecture

The iMFMLPDenoiser has a **shared backbone** plus two separate output heads (u-head for average velocity, v-head for instantaneous velocity), all conditioned on $t$ and $h = t - r$.

```
iMFMLPDenoiser:  (z_t ∈ R², t ∈ R, h ∈ R)  →  (u ∈ R², v ∈ R²)

Conditioning:  c = h_embed(h) + t_embed(t)

                              ┌─ [AdaLN ResBlock × 4] → u-head → u
z_t → Linear → [Shared × 2] ─┤
                              └─ [AdaLN ResBlock × 4] → v-head → v
```

```python
# imf_mlp.py: iMFMLPDenoiser
def forward(self, x, t, h):
    feat = self.input_proj(x)
    c = self.h_embed(h) + self.t_embed(t)

    for block in self.shared_blocks:                 # shared backbone
        feat = block(feat, c)

    u = feat                                         # u-head
    for block in self.u_blocks:
        u = block(u, c)
    u = self.u_proj(u)

    v = feat                                         # v-head
    for block in self.v_blocks:
        v = block(v, c)
    v = self.v_proj(v)

    return u, v
```

**2D-specific note**: The reference iMF conditions only on $h$, because in high-dimensional data (images) the noise level can be inferred from $z_t$. For 2D data, explicit $t$ conditioning is necessary since low-dimensional $z_t$ doesn't carry enough information to infer the noise level.

---

## Training Loss

The loss regresses the compound function $V$ toward the fixed target $v_{\text{target}} = \epsilon - x_0$, plus an auxiliary v-head loss:

1. **Sample time pairs**: $t, r$ from logit-normal, with $t \geq r$. 50% of the batch uses $r = t$ (flow matching regime).

2. **Construct noisy sample and target**:

$$z_t = (1-t) x_0 + t \epsilon, \quad v_{\text{target}} = \epsilon - x_0$$

3. **Get detached instantaneous velocity** from the v-head at $h = 0$:

```python
# imf.py: training_loss()
with torch.no_grad():
    _, v_c = model(z_t, t, h=0)                     # v-head estimate (detached)
```

4. **Compute $u$, $v$, and $du/dt$ via JVP**: The tangent vector uses $v_c$ (detached v-head output) as the direction for differentiating through $z_t$:

```python
# imf.py: training_loss()
u, du_dt, v = torch.func.jvp(
    u_fn, (z_t, t, r), (v_c, ones, zeros), has_aux=True
)
```

5. **Compound function and loss**:

$$V = u + (t - r) \cdot \text{sg}(du/dt)$$

$$\mathcal{L} = \|V - v_{\text{target}}\|^2 + \|v - v_{\text{target}}\|^2$$

Both terms use adaptive weighting to prevent high-loss samples from dominating.

```python
# imf.py: training_loss()
V = u + (t - r) * du_dt.detach()                    # compound function

loss_u = adaptive_weight((V - v_target).pow(2).sum(-1))
loss_v = adaptive_weight((v - v_target).pow(2).sum(-1))
loss = (loss_u + loss_v).mean()
```

---

## Sampling

Sampling is identical to MeanFlow — Euler integration using the average velocity $u$ (the v-head is not used at inference):

### One-Step (num_steps=1)

$$x_0 = z - u(z, t=1, h=1), \quad z \sim \mathcal{N}(0, I)$$

### Multi-Step Euler

```python
# imf.py: sample()
z = torch.randn(shape, device=device)
t_steps = linspace(1.0, 0.0, num_steps + 1)

for i in range(num_steps):
    t, r = t_steps[i], t_steps[i + 1]
    h = t - r
    u, _ = model(z, t, h)                           # only use u-head
    z = z - h * u
```

Multi-step sampling (e.g., 100 steps) generally produces better results than single-step, as one-step generation with $h = 1$ can be out-of-distribution for the logit-normal training distribution.

---

## Connection to MeanFlow

Both MeanFlow and iMF learn the same average velocity field and sample identically. The difference is purely in the training loss formulation. MeanFlow constructs a model-dependent target $u_{\text{tgt}} = v - (t-r) \cdot \text{sg}(du/dt)$ and regresses $u$ toward it. iMF instead constructs the compound function $V = u + (t-r) \cdot \text{sg}(du/dt)$ and regresses it toward the fixed target $\epsilon - x_0$. This makes iMF's training more stable because the target doesn't depend on the model's current predictions. The dual-head architecture further helps by providing a dedicated v-head for the JVP computation.
