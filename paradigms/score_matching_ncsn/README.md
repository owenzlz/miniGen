# Score Matching (NCSN)

> Song & Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution", NeurIPS 2019.

NCSN (Noise Conditional Score Network) learns the **score function** $\nabla_x \log p(x)$ — the gradient of the log-density — at multiple noise levels. Samples are generated via **Annealed Langevin Dynamics**, iterating from high noise to low noise.

**Intuition**: Hyv&auml;rinen's score matching works on clean data but struggles in practice — the score is only well-defined where data actually lives, and in low-density regions (most of the space) the model gets no training signal. NCSN fixes this by training at multiple noise levels: at high noise, the data distribution is spread out and the score is well-behaved everywhere; at low noise, the score captures fine details. The key idea is that learning to denoise is equivalent to learning the score — if you know what noise was added, you implicitly know which direction points toward clean data. At sampling time, we start from pure noise and progressively denoise by following the learned score at decreasing noise levels.

---

## Noise Levels

A geometric sequence of $L$ noise levels from large to small:

$$\sigma_0 > \sigma_1 > \cdots > \sigma_{L-1}, \quad \sigma_i = \sigma_{\max} \cdot \left(\frac{\sigma_{\min}}{\sigma_{\max}}\right)^{i/(L-1)}$$

```python
# score_matching_ncsn.py: __init__()
self.sigmas = torch.exp(
    torch.linspace(math.log(sigma_max), math.log(sigma_min), num_noise_levels)
)
```

Default config: $L = 10$ levels from $\sigma_{\max} = 1.0$ down to $\sigma_{\min} = 0.01$.

---

## Loss Derivation

### Step 1: The Ideal Objective

Like Hyv&auml;rinen score matching, we want $s_\theta(x, \sigma) \approx \nabla_x \log p_\sigma(x)$, where $p_\sigma$ is the data distribution convolved with Gaussian noise of scale $\sigma$. The ideal loss is:

$$J(\theta) = \frac{1}{2} \sum_{i=0}^{L-1} \mathbb{E}_{p_{\sigma_i}(x)} \bigl[ \lVert s_\theta(x, i) - \nabla_x \log p_{\sigma_i}(x) \rVert^2 \bigr]$$

**Problem**: we still don't know the true score $\nabla_x \log p_{\sigma_i}(x)$.

### Step 2: Denoising Score Matching (Vincent, 2011)

The key insight: for a noisy sample $\tilde{x} = x_0 + \sigma \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$, the score of the noise-perturbed distribution has a simple closed form:

$$\nabla_{\tilde{x}} \log p(\tilde{x} \mid x_0) = \nabla_{\tilde{x}} \log \mathcal{N}(\tilde{x}; x_0, \sigma^2 I) = -\frac{\tilde{x} - x_0}{\sigma^2} = -\frac{\epsilon}{\sigma}$$

Vincent (2011) proved that matching the score of the **conditional** $p(\tilde{x} \mid x_0)$ in expectation over $x_0$ is equivalent to matching the score of the **marginal** $p_\sigma(\tilde{x})$:

$$\mathbb{E}_{p_\sigma(\tilde{x})} \bigl[ \lVert s_\theta - \nabla_{\tilde{x}} \log p_\sigma \rVert^2 \bigr] = \mathbb{E}_{x_0, \epsilon} \bigl[ \lVert s_\theta(\tilde{x}, \sigma) - (-\epsilon / \sigma) \rVert^2 \bigr] + \text{const}$$

This eliminates the unknown marginal score entirely.

### Step 3: Noise Prediction Parameterization

Instead of predicting the score directly, the model predicts the **noise** $\epsilon$. Since $\text{score} = -\epsilon / \sigma$, the two are equivalent up to a scaling factor. This gives the final loss:

$$\boxed{\mathcal{L} = \mathbb{E}_{i,\, x_0,\, \epsilon} \bigl[ \lVert s_\theta(x_0 + \sigma_i \epsilon,\; i) - \epsilon \rVert^2 \bigr]}$$

This is a simple MSE between predicted and actual noise — no Jacobian trace, no integration by parts, just a forward pass and an L2 loss. This is why denoising score matching scales to high dimensions while Hyv&auml;rinen's original method does not.

---

## Training

```python
# score_matching_ncsn.py: training_loss()
sigma = sigmas[t]                              # σ_i for each sample in batch
noise = torch.randn_like(x0)                  # ε ~ N(0, I)
x_noisy = x0 + sigma.unsqueeze(-1) * noise    # x̃ = x₀ + σ_i ε

model_out = model(x_noisy, t)                 # predict ε
loss = F.mse_loss(model_out, noise)            # ‖s_θ(x̃, i) - ε‖²
```

Each training step: sample a random noise level index $i$, perturb clean data by $\sigma_i$, and regress the noise. The noise level index $i$ is passed to the model as conditioning (analogous to the timestep in diffusion models).

---

## Sampling: Annealed Langevin Dynamics

### Langevin Dynamics

Given a score function, Langevin dynamics generates samples by iteratively following the score with injected noise:

$$x_{k+1} = x_k + \frac{\alpha}{2} \nabla_x \log p(x_k) + \sqrt{\alpha} \, z_k, \quad z_k \sim \mathcal{N}(0, I)$$

As $\alpha \to 0$ and $k \to \infty$, the samples converge to the true distribution $p(x)$.

### Annealing Across Noise Levels

Plain Langevin from random noise struggles to find modes of a complex distribution. Annealed Langevin solves this by iterating through noise levels from high to low, running multiple Langevin steps at each level:

- **High noise** ($\sigma_0$): the distribution is spread out, score is smooth, Langevin can explore broadly.
- **Low noise** ($\sigma_{L-1}$): the distribution is sharp, score captures fine details, Langevin refines.

The step size is adapted to each noise level:

$$\alpha_i = \epsilon \cdot \left(\frac{\sigma_i}{\sigma_{L-1}}\right)^2$$

Since the model predicts noise $\epsilon$ (not score), the score is recovered as $\nabla_x \log p_{\sigma_i}(x) = -s_\theta(x, i) / \sigma_i$.

```python
# score_matching_ncsn.py: sample()
x = torch.randn(shape, device=device) * sigmas[0]      # init from large noise

for i in range(num_noise_levels):
    sigma = sigmas[i]
    alpha = eps * (sigma / sigmas[-1]) ** 2              # adaptive step size

    for _ in range(steps_per_level):
        noise_pred = model(x, t_batch)
        score = -noise_pred / sigma                      # score = -ε/σ
        z = torch.randn_like(x)
        x = x + alpha / 2 * score + sqrt(alpha) * z     # Langevin update
```

---

## Connection to Diffusion Models

Score matching and DDPM are closely related (unified by Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations", ICLR 2021):

| | NCSN | DDPM |
|---|---|---|
| SDE type | **Variance Exploding (VE)** | **Variance Preserving (VP)** |
| Forward process | $\tilde{x} = x_0 + \sigma \epsilon$ (add noise, variance grows) | $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$ (scale down + add noise, variance bounded) |
| Noise levels | Geometric $\sigma_i$ | Schedule $\beta_t \to \bar{\alpha}_t$ |
| Network predicts | Noise $\epsilon$ ($= -\sigma \cdot \text{score}$) | Noise $\epsilon$ |
| Sampling | Langevin dynamics | Ancestral sampling |
