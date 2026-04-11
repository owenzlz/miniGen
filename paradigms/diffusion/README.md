# Diffusion (DDPM)

> Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.

Diffusion models learn to reverse a gradual noising process. The **forward process** adds Gaussian noise over $T$ timesteps until data becomes pure noise. A neural network is trained to **reverse** this process, denoising step by step. Diffusion models are scretely score-based models, as its prediction mathematically approximates "score" - log likelihood of data distribution. 

**Intuition**: Imagine slowly adding static to a photograph until it becomes pure white noise — this is easy and requires no learning. Now imagine training a neural network to undo one tiny step of that corruption: given a slightly noisy image, predict what it looked like one step earlier. If the network can do this well at every noise level, we can start from pure noise and iteratively denoise to produce a clean sample. The key insight is that each denoising step is a simple regression problem (predict the noise that was added), and the chain of small steps composes into a powerful generative model. This is the VP (Variance Preserving) SDE — the forward process scales down the signal while adding noise, keeping the total variance bounded.

---

## Forward Process

The forward process $q(x_t | x_0)$ adds noise to clean data $x_0$:

$$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

where $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$ is the cumulative product of the noise schedule.

Using the **reparameterization trick**, we can sample $x_t$ directly:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

```python
# ddpm.py: q_sample()
sqrt_alpha_cumprod = extract(self.schedule.sqrt_alphas_cumprod, t, x0.shape)
sqrt_one_minus_alpha_cumprod = extract(self.schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape)

xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
```

---

## Noise Schedules

The schedule defines $\beta_t$ (per-step noise variance), from which all other quantities are derived.

### Linear Schedule (Ho et al., 2020)

$$\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})$$

```python
# scheduler/linear.py
betas = torch.linspace(beta_start, beta_end, num_timesteps)
```

### Cosine Schedule (Nichol & Dhariwal, 2021)

Defines $\bar{\alpha}_t$ via a cosine function for gentler noise addition:

$$f(t) = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right), \quad \bar{\alpha}_t = \frac{f(t)}{f(0)}$$

Then $\beta_t$ is derived: $\beta_t = 1 - \bar{\alpha}_t / \bar{\alpha}_{t-1}$.

```python
# scheduler/cosine.py
f_t = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
alphas_cumprod = f_t / f_t[0]
betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
```

### Derived Quantities

From $\beta_t$, the schedule precomputes all buffers:

| Symbol | Formula | Buffer Name |
|--------|---------|-------------|
| $\alpha_t$ | $1 - \beta_t$ | `alphas` |
| $\bar{\alpha}_t$ | $\prod_{s=1}^{t} \alpha_s$ | `alphas_cumprod` |
| $\sqrt{\bar{\alpha}_t}$ | — | `sqrt_alphas_cumprod` |
| $\sqrt{1-\bar{\alpha}_t}$ | — | `sqrt_one_minus_alphas_cumprod` |
| $\tilde{\beta}_t$ | $\frac{(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_t)} \beta_t$ | `posterior_variance` |

```python
# scheduler/base.py: _compute_buffers()
alphas = 1.0 - betas                                    # α_t
alphas_cumprod = torch.cumprod(alphas, dim=0)            # ᾱ_t
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)         # √ᾱ_t
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)  # √(1-ᾱ_t)

# Posterior variance: β̃_t
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
```

---

## Training Loss

The network $\epsilon_\theta(x_t, t)$ is trained to predict the noise $\epsilon$ added during the forward process:

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \lVert \epsilon_\theta(x_t, t) - \epsilon \rVert^2 \right]$$

```python
# ddpm.py: training_loss()
noise = torch.randn_like(x0)
xt, _ = self.q_sample(x0, t, noise)
model_out = model(xt, t, **kwargs)
target = pred.get_training_target(x0, noise, t, self.parameterization, ...)
loss = F.mse_loss(model_out, target)
```

### Parameterization Variants

The network can predict different targets. All are mathematically equivalent but differ in training dynamics:

| Parameterization | Target | Recover $x_0$ via |
|-----------------|--------|-------------------|
| $\epsilon$-prediction | $\epsilon$ | $\hat{x}_0 = (x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta) / \sqrt{\bar{\alpha}_t}$ |
| $x_0$-prediction | $x_0$ | $\hat{x}_0 = \text{model output}$ |
| $v$-prediction | $v = \sqrt{\bar{\alpha}_t} \epsilon - \sqrt{1-\bar{\alpha}_t} x_0$ | $\hat{x}_0 = \sqrt{\bar{\alpha}_t} x_t - \sqrt{1-\bar{\alpha}_t} v$ |

```python
# predict_parameterization.py: model_output_to_x0()
if parameterization == EPS:
    return (xt - sqrt_one_minus_alpha * eps) / sqrt_alpha
elif parameterization == X0:
    return model_output
elif parameterization == V:
    return sqrt_alpha * xt - sqrt_one_minus_alpha * v
```

---

## Reverse Process (Sampling)

### DDPM Sampler (Algorithm 2, Ho et al.)

The reverse process computes the posterior $q(x_{t-1} \mid x_t, x_0)$:

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t, \tilde{\beta}_t \mathbf{I})$$

where the posterior mean is:

$$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

Each reverse step samples:

$$x_{t-1} = \tilde{\mu}_\theta(x_t, t) + \sqrt{\tilde{\beta}_t} z, \quad z \sim \mathcal{N}(0, \mathbf{I})$$

```python
# sampler/ddpm_sampler.py: p_sample()
mean, _, log_variance = self.p_mean_variance(model, xt, t, **kwargs)
noise = torch.randn_like(xt)
nonzero_mask = (t != 0).float().view(...)    # No noise at t=0
return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise
```

The full sampling loop iterates from $t = T-1$ down to $t = 0$:

```python
# sampler/ddpm_sampler.py: sample()
x = torch.randn(shape, device=device)        # x_T ~ N(0, I)
for t in reversed(range(T)):
    x = self.p_sample(model, x, t_batch)
```

### DDIM Sampler (Song et al., 2021)

DDIM enables **fewer steps** and **deterministic** sampling via:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon_\theta + \sigma_t \epsilon$$

where $\sigma_t = \eta \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}} \sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$ controls stochasticity:
- $\eta = 0$: deterministic (DDIM)
- $\eta = 1$: stochastic (recovers DDPM)

```python
# sampler/ddim_sampler.py: p_sample()
x0_pred = pred.model_output_to_x0(model_out, xt, t, ...)
eps_pred = pred.predict_eps_from_x0(xt, t, x0_pred, ...)

sigma = self.eta * torch.sqrt(
    (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) *
    (1 - alpha_cumprod_t / alpha_cumprod_prev)
)
pred_dir = torch.sqrt(1 - alpha_cumprod_prev - sigma ** 2) * eps_pred
x_prev = torch.sqrt(alpha_cumprod_prev) * x0_pred + pred_dir + sigma * noise
```

DDIM uses a **subsequence** of timesteps for accelerated sampling:

```python
# sampler/ddim_sampler.py: _get_timestep_schedule()
step_ratio = T // num_steps
timesteps = torch.arange(0, num_steps) * step_ratio  # Evenly spaced subset
```

---

## Classifier-Free Guidance (CFG)

> Ho & Salimans, "Classifier-Free Diffusion Guidance", NeurIPS 2021.

CFG steers generation toward a class $y$ without a separate classifier:

$$\epsilon_{\text{cfg}} = \epsilon_\theta(x_t, t, \varnothing) + s \cdot (\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \varnothing))$$

where $s > 1$ amplifies the class signal, and $\varnothing$ is the null (unconditional) token.

**Training**: Randomly drop labels with probability $p$ (replace with null class).

```python
# models/mlp.py: _drop_labels()
drop_mask = torch.rand(labels.shape[0]) < self.class_dropout_prob
labels = torch.where(drop_mask, self.num_classes, labels)
```

**Inference**: Combine conditional and unconditional predictions.

```python
# guidance.py: CFGWrapper.forward()
out_uncond = self.model(x, t, y=y_null, **kwargs)
out_cond = self.model(x, t, y=y, **kwargs)
return out_uncond + self.cfg_scale * (out_cond - out_uncond)
```
