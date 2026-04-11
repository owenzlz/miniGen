# Consistency Training (iCT)

> Song et al., "Consistency Models", ICML 2023.
> Song & Dhariwal, "Improved Techniques for Training Consistency Models", ICLR 2024 (Oral).

A consistency model learns to map **any point on a noise trajectory directly to the clean data** in a single step. This implementation uses **improved Consistency Training (iCT)** — training from scratch via a self-consistency objective, with no pre-trained diffusion model needed.

**Intuition**: In a diffusion model, denoising from noise to data requires many small steps along a trajectory. A consistency model learns a shortcut: given any noisy version of the data, jump directly to the clean output. The key constraint is *self-consistency* — two points on the same trajectory (same data, same noise direction, different noise levels) must map to the same clean output. Training enforces this by taking the same noise $z$, creating two versions at adjacent noise levels $\sigma_n$ and $\sigma_{n+1}$, and requiring the model's outputs to match. The stop-gradient target prevents collapse, and a progressive schedule gradually increases the number of discretization steps during training.

---

## Model Architecture

Consistency training reuses the standard MLPDenoiser, but wraps it with **EDM-style preconditioning** — learned skip connections that depend on the noise level $\sigma$.

```
Raw network:   (c_in · x, σ)  →  MLPDenoiser  →  F_θ

Preconditioned output:  f(x, σ) = c_skip · x + c_out · F_θ(c_in · x, σ)
```

The preconditioning coefficients ensure that at $\sigma = 0$ the model outputs $x$ unchanged (identity), and at large $\sigma$ the model output dominates:

$$c_{\text{skip}} = \frac{\sigma_d^2}{\sigma^2 + \sigma_d^2}, \quad c_{\text{out}} = \frac{\sigma \cdot \sigma_d}{\sqrt{\sigma^2 + \sigma_d^2}}, \quad c_{\text{in}} = \frac{1}{\sqrt{\sigma^2 + \sigma_d^2}}$$

where $\sigma_d = 0.5$ is the data standard deviation.

```python
# consistency_training.py: _consistency_fn()
c_skip = sd2 / (s2 + sd2)
c_out = sigma * sigma_data / sqrt(s2 + sd2)
c_in = 1.0 / sqrt(s2 + sd2)

model_out = model(c_in * x, sigma)
return c_skip * x + c_out * model_out
```

---

## Karras Sigma Schedule

Noise levels are discretized using the Karras schedule, which spaces $N$ sigma values between $\sigma_{\min}$ and $\sigma_{\max}$ with denser spacing at lower noise levels:

$$\sigma_i = \left(\sigma_{\min}^{1/\rho} + \frac{i}{N-1}\left(\sigma_{\max}^{1/\rho} - \sigma_{\min}^{1/\rho}\right)\right)^\rho$$

with $\rho = 7$, $\sigma_{\min} = 0.002$, $\sigma_{\max} = 1.0$.

```python
# consistency_training.py: _karras_sigmas()
indices = torch.arange(N, device=device)
sigmas = (min_inv + indices / (N - 1) * (max_inv - min_inv)) ** self.rho
```

---

## Progressive Schedule

The number of discretization steps $N$ increases over training, starting coarse and ending fine:

$$N(k) = \left\lceil\sqrt{\frac{k}{K}\left((s_1+1)^2 - s_0^2\right) + s_0^2} - 1\right\rceil + 1$$

where $k$ is the current step, $K$ is total training steps, $s_0 = 10$, and $s_1 = 150$. Early in training, adjacent sigma pairs are far apart (easy consistency), and as training progresses, they get closer (harder, finer consistency).

```python
# consistency_training.py: _current_num_timesteps()
ratio = k / K
N = ceil(sqrt(ratio * ((s1 + 1)**2 - s0**2) + s0**2) - 1) + 1
```

---

## Training Loss

For each training step:

1. **Sample adjacent sigma pairs** $(\sigma_n, \sigma_{n+1})$ from the current Karras schedule, weighted by a lognormal distribution (biases toward medium noise levels).

2. **Create noisy pairs** with the same noise direction but different magnitudes:

$$x_{n+1} = x_0 + \sigma_{n+1} z, \quad x_n = x_0 + \sigma_n z, \quad z \sim \mathcal{N}(0, I)$$

3. **Compute consistency outputs** — online (with gradient) and target (stop-gradient):

$$\text{online} = f_\theta(x_{n+1}, \sigma_{n+1}), \quad \text{target} = \text{sg}(f_\theta(x_n, \sigma_n))$$

4. **Pseudo-Huber loss** with inverse step-size weighting:

$$\mathcal{L} = \frac{1}{\sigma_{n+1} - \sigma_n} \left(\sqrt{\|\text{online} - \text{target}\|^2 + c^2} - c\right)$$

where $c = 0.00054 \sqrt{d}$. The inverse weighting upweights pairs with small sigma gaps (fine consistency), which become more important as training progresses.

```python
# consistency_training.py: training_loss()
z = torch.randn_like(x0)
x_n1 = x0 + sigma_n1 * z                             # noisier
x_n = x0 + sigma_n * z                                # less noisy (same z!)

online = self._consistency_fn(model, x_n1, sigma_n1)  # with gradient
with torch.no_grad():
    target = self._consistency_fn(model, x_n, sigma_n) # stop-gradient

lam = 1.0 / (sigma_n1 - sigma_n)                      # inverse step-size weight
loss = (lam * pseudo_huber(online, target)).mean()
```

---

## Sampling

### Single-Step (num_steps=1)

Sample noise at $\sigma_{\max}$ and apply the consistency function once:

$$x \sim \mathcal{N}(0, \sigma_{\max}^2 I), \quad \hat{x}_0 = f_\theta(x, \sigma_{\max})$$

```python
# consistency_training.py: sample()
x = torch.randn(shape) * self.sigma_max
x = self._consistency_fn(model, x, sigma_max)
```

### Multi-Step Refinement (num_steps > 1)

Iteratively denoise and re-noise at decreasing sigma levels for higher quality:

```python
# consistency_training.py: sample() — multi-step
x = self._consistency_fn(model, x, sigma_max)          # initial denoise

for sigma_i in sigmas[:-1]:                            # decreasing sigmas
    x = x + sqrt(sigma_i**2 - sigma_min**2) * z       # re-noise
    x = self._consistency_fn(model, x, sigma_i)        # denoise again
```

Each re-noise step adds less noise than the previous one, so the model progressively refines the sample.
