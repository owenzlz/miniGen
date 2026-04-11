# Score Matching (Hyv&auml;rinen)

> Hyv&auml;rinen, "Estimation of Non-Normalized Statistical Models by Score Matching", JMLR 2005.

Score matching learns the **score function** $s_\theta(x) \approx \nabla_x \log p(x)$ — the gradient of the log-density — directly on clean data, without ever needing to know $p(x)$ itself.

**Intuition**: If we knew the true score $\nabla_x \log p(x)$ at every point, we could train a network to match it with a simple MSE loss. But we don't know the true score, which is what we need to learn. Hyv&auml;rinen's key insight is that through integration by parts, we can rewrite this MSE objective into an equivalent form that only involves the model's own outputs and their derivatives — no true score needed. The resulting loss has two terms: one measuring how much the model's score field "diverges" (Jacobian trace), and one penalizing large score magnitudes.

---

## Loss Derivation

### Step 1: The Ideal Objective

We want our model score $s_\theta(x)$ to match the true score $\nabla_x \log p(x)$. The natural objective is:

$$J(\theta) = \frac{1}{2} \mathbb{E}_{p(x)} \bigl[ \lVert s_\theta(x) - \nabla_x \log p(x) \rVert^2 \bigr]$$

**Problem**: this requires the unknown true score $\nabla_x \log p(x)$.

### Step 2: Expand the Square

Expanding $\lVert a - b \rVert^2 = \lVert a \rVert^2 - 2a^\top b + \lVert b \rVert^2$:

$$J(\theta) = \mathbb{E}_{p(x)} \bigl[ \underbrace{\tfrac{1}{2} \lVert s_\theta(x) \rVert^2}_{\text{(A) model norm}} - \underbrace{s_\theta(x)^\top \nabla_x \log p(x)}_{\text{(B) cross term}} + \underbrace{\tfrac{1}{2} \lVert \nabla_x \log p(x) \rVert^2}_{\text{(C) constant}} \bigr]$$

- **(A)** is computable — it only involves the model.
- **(C)** is a constant w.r.t. $\theta$ — we can drop it for optimization.
- **(B)** still contains the unknown true score. This is the term we need to eliminate.

### Step 3: Integration by Parts (Hyv&auml;rinen's Trick)

The cross term (B) can be rewritten. Consider one dimension $d$:

$$\mathbb{E}_{p(x)} \bigl[ [s_\theta(x)]_d \cdot \partial_d \log p(x) \bigr] = \int [s_\theta(x)]_d \cdot \partial_d \log p(x) \cdot p(x) \, dx$$

Since $\partial_d \log p(x) \cdot p(x) = \partial_d p(x)$, this becomes:

$$= \int [s_\theta(x)]_d \cdot \partial_d p(x) \, dx$$

Applying integration by parts ($\int u \, dv = uv - \int v \, du$), with the boundary term vanishing (assuming $p(x) \to 0$ at infinity):

$$= -\int \partial_d [s_\theta(x)]_d \cdot p(x) \, dx = -\mathbb{E}_{p(x)} \bigl[ \partial_d [s_\theta(x)]_d \bigr]$$

Summing over all dimensions $d$:

$$-\mathbb{E}_{p(x)} \bigl[ s_\theta(x)^\top \nabla_x \log p(x) \bigr] = \mathbb{E}_{p(x)} \Bigl[ \sum_d \partial_d [s_\theta(x)]_d \Bigr] = \mathbb{E}_{p(x)} \bigl[ \text{Tr}(\nabla_x s_\theta(x)) \bigr]$$

### Step 4: The Final Loss

Substituting back and dropping the constant (C):

$$\boxed{\mathcal{L}(\theta) = \mathbb{E}_{p(x)} \bigl[ \text{Tr}(\nabla_x s_\theta(x)) + \tfrac{1}{2} \lVert s_\theta(x) \rVert^2 \bigr]}$$

This loss is **equivalent** to the original MSE objective (up to a constant) but is fully computable — it only requires the model output and its Jacobian, not the true score.

---

## Training

The Jacobian trace $\text{Tr}(\partial s_\theta / \partial x) = \sum_d \partial [s_\theta(x)]_d / \partial x_d$ is computed via autograd. For 2D data, this is just two autograd calls:

```python
# score_matching_hyvarinen.py: _jacobian_trace_2d()
trace = torch.zeros(B, device=device)
for d in range(2):
    grad_d = torch.autograd.grad(s[:, d].sum(), x, create_graph=True)[0]
    trace = trace + grad_d[:, d]   # ∂s_d/∂x_d
```

The full training loss:

```python
# score_matching_hyvarinen.py: training_loss()
x = x0.detach().requires_grad_(True)         # enable grad for Jacobian

score = model(x, t)                           # s_θ(x)
trace = _jacobian_trace_2d(score, x)          # Tr(∂s_θ/∂x)
score_norm_sq = 0.5 * (score ** 2).sum(-1)    # ½‖s_θ(x)‖²

loss = (trace + score_norm_sq).mean()
```

### Why the Loss is Negative

At the optimum $s_\theta = \nabla_x \log p$, the trace becomes the Laplacian $\Delta \log p$. By a second integration by parts, $\mathbb{E}_p[\Delta \log p] = -\mathbb{E}_p[\lVert \nabla_x \log p \rVert^2]$, so:

$$\mathcal{L}^* = -\tfrac{1}{2} \mathbb{E}_{p(x)} \bigl[ \lVert \nabla_x \log p(x) \rVert^2 \bigr] \leq 0$$

The optimal loss is always negative. For distributions concentrated on thin manifolds (like swiss roll), score norms are large, producing a large negative loss. This is expected and correct.

---

## Sampling: Langevin Dynamics

Since the model predicts the score directly (no noise levels), sampling uses plain Langevin dynamics:

$$x_{k+1} = x_k + \frac{\epsilon}{2} s_\theta(x_k) + \sqrt{\epsilon} \, z_k, \quad z_k \sim \mathcal{N}(0, I)$$

```python
# score_matching_hyvarinen.py: sample()
x = torch.randn(shape, device=device)

for _ in range(steps):
    score = model(x, t_batch)
    x = x + (eps / 2) * score + sqrt(eps) * randn_like(x)
```

Unlike NCSN, there is no annealing across noise levels — the model learns a single score function at the data distribution's noise level (zero).

---

## Comparison with NCSN (Denoising Score Matching)

| | Hyv&auml;rinen (2005) | NCSN (Song & Ermon, 2019) |
|---|---|---|
| Training data | Clean $x$ | Noisy $x + \sigma_i \epsilon$ |
| Loss | $\text{Tr}(\nabla_x s_\theta) + \frac{1}{2} \lVert s_\theta \rVert^2$ | $\lVert s_\theta(x + \sigma\epsilon, \sigma) + \epsilon / \sigma \rVert^2$ |
| Requires | Jacobian trace (2nd-order autograd) | Only 1st-order forward pass |
| Noise levels | None (single score function) | Multiple $\sigma_i$ (geometric schedule) |
| Sampling | Plain Langevin | Annealed Langevin |
| Loss sign | Negative at optimum | Always non-negative (MSE) |
| Scalability | $O(D)$ autograd calls for trace | Scales to any dimension |

Hyv&auml;rinen's method is the theoretical foundation, but the Jacobian trace makes it expensive for high-dimensional data. Denoising score matching (Vincent 2011, used by NCSN) avoids this by adding noise and regressing it, which is why it became the dominant approach for images.
