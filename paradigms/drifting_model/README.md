# Drifting Model

> Deng et al., "Generative Modeling via Drifting", 2026.

A drifting model trains a generator by evolving the pushforward distribution toward the data distribution during training. At each step, a **drifting field** $V$ nudges generated samples toward real data (attraction) and away from other generated samples (repulsion). The generator is updated to match these nudged positions. At equilibrium ($q = p$), the drift vanishes and sampling is a **single forward pass**.

**Intuition**: Unlike flow matching or diffusion, which learn to reverse a noise-corruption process, a drifting model starts with a generator that outputs garbage and gradually improves it by telling each generated sample where it *should* move. The drifting field acts like a gravitational force: real data points attract nearby generated samples, while other generated samples repel each other to prevent mode collapse. The generator learns to produce samples that already sit at the right locations — no denoising or ODE integration needed at inference, just one forward pass through the network.

---

## Model Architecture

The generator is a simple MLP that maps latent noise to data space — no time conditioning, no skip connections. Unlike the GAN generator (which uses tanh output), the drifting generator has **no output activation**.

```
DriftingGenerator:  z ~ N(0, I) ∈ R^32  →  Linear(32→256) → SiLU → [Linear(256→256) → SiLU] × 2 → Linear(256→2)  →  x ∈ R²
```

```python
# drifting_mlp.py: DriftingGenerator
layers = [Linear(latent_dim, hidden_size), SiLU()]
for _ in range(depth - 1):
    layers.extend([Linear(hidden_size, hidden_size), SiLU()])
layers.append(Linear(hidden_size, output_dim))          # no output activation
self.net = nn.Sequential(*layers)

def forward(self, z):
    return self.net(z)                                   # z → x
```

---

## The Drifting Field

The drifting field $V(x)$ is the difference between an attraction term (toward data) and a repulsion term (away from other generated samples):

$$V(x) = V^+(x) - V^-(x)$$

Both terms use an **exponential kernel** $k(x, y) = \exp(-\|x - y\| / \tau)$ with temperature $\tau = 0.05$:

- **$V^+$ (attraction)**: weighted mean-shift toward real data points, pulling generated samples toward nearby data.
- **$V^-$ (repulsion)**: weighted mean-shift toward other generated samples, pushing them apart to cover all modes.

The kernel is **doubly normalized** (along both rows and columns) for numerical stability.

```python
# drifting.py: compute_drift()
# gen: [G, D] generated samples, pos: [P, D] real data samples

# ── Step 1: Build kernel matrix ────────────────────────────────
targets = cat([gen, pos])                               # [G+P, D]
dist = torch.cdist(gen, targets)                        # [G, G+P] pairwise distances
dist[:, :G].fill_diagonal_(1e6)                         # mask self (gen_i → gen_i)
kernel = (-dist / self.temp).exp()                      # [G, G+P] exponential kernel
```

The kernel matrix has two blocks:

| | gen columns (`:G`) | data columns (`G:`) |
|---|---|---|
| **gen rows** | $K_{gg}$ — gen-to-gen affinities | $K_{gd}$ — gen-to-data affinities |

Self-distances on the diagonal of $K_{gg}$ are masked to $10^6$ so $k(x_i, x_i) \approx 0$, preventing a point from attracting itself.

```python
# ── Step 2: Doubly normalize (Sinkhorn-like, 1 iteration) ─────
# Divide each entry by sqrt(row_sum × col_sum)
normalizer = sqrt(kernel.sum(dim=-1) * kernel.sum(dim=-2))   # [G, G+P]
K = kernel / normalizer
```

This makes the kernel matrix approximately doubly stochastic — entries are balanced so that no single row or column dominates. Without this, a generated point near many data points would receive disproportionately large drift.

```python
# ── Step 3: Cross-weighted attraction and repulsion ────────────
# For each generated point i, define:
#   s_gg(i) = Σ_k K_gg[i,k]     (total gen-gen affinity — how crowded)
#   s_gd(i) = Σ_j K_gd[i,j]     (total gen-data affinity — how close to data)

pos_coeff = K[:, G:] * K[:, :G].sum(dim=-1, keepdim=True)   # [G, P]
neg_coeff = K[:, :G] * K[:, G:].sum(dim=-1, keepdim=True)   # [G, G]

V = pos_coeff @ pos - neg_coeff @ gen                        # [G, D]
```

The coefficient construction uses **cross-weighting**: each gen-data pair $(i, j)$ in the attraction term is scaled by $s_{gg}(i)$, and each gen-gen pair $(i, k)$ in the repulsion term is scaled by $s_{gd}(i)$:

$$V_i = \underbrace{s_{gg}(i) \sum_j K_{gd}(i,j) \cdot x_j^{\text{data}}}_{\text{attraction}} \;-\; \underbrace{s_{gd}(i) \sum_k K_{gg}(i,k) \cdot x_k^{\text{gen}}}_{\text{repulsion}}$$

The intuition: when a generated point is in a **crowded area** (high $s_{gg}$), its attraction toward data is amplified — it should escape the crowd. When it's **near data** (high $s_{gd}$), its repulsion from other generated points is amplified — it should spread out to cover more modes. At equilibrium ($q = p$), the two terms cancel exactly and $V = 0$ everywhere, meaning the generator already produces the correct distribution.

---

## Training

Training is remarkably simple — no adversarial dynamics, no noise schedule, no ODE:

1. **Generate samples**: $\hat{x} = G(z), \quad z \sim \mathcal{N}(0, I)$
2. **Compute drift** $V$ (detached — no gradient through the kernel):

$$\text{target} = \text{sg}(\hat{x} + V(\hat{x}, x_{\text{data}}))$$

3. **Update generator** to match the nudged positions:

$$\mathcal{L} = \|\hat{x} - \text{target}\|^2$$

```python
# drifting.py: training_loss()
z = torch.randn(B, self.latent_dim, device=device)
gen = model(z)                                          # generate

with torch.no_grad():
    V = self.compute_drift(gen, x0)                     # drifting field
    target = (gen + V).detach()                         # nudged positions

return F.mse_loss(gen, target)                          # generator follows
```

Gradients only flow through the generator — the drifting field computation is fully detached.

### Training Dynamics

The loss $\|V\|^2$ is inherently tiny (1e-6 to 1e-4) because the drift field produces small corrections. The generator bootstraps from near-zero output and slowly expands; the exponential kernel ($\tau = 0.05$) only activates once generated samples are within ~0.25 of real data points. Large batch sizes (2048) help by providing denser coverage for the kernel computation.

---

## Sampling

Sampling is a single forward pass — the simplest of any generative model:

$$z \sim \mathcal{N}(0, I), \quad x = G(z)$$

```python
# drifting.py: sample()
z = torch.randn(num_samples, self.latent_dim, device=device)
return model(z)
```

No iterative denoising, no ODE integration, no MCMC — just one network evaluation.

---

## Connection to GANs

Like a GAN, the drifting model uses a generator that maps noise to data in one step. But the training signal is completely different. A GAN trains via an adversarial discriminator (minimax game, mode collapse risk, training instability). A drifting model uses a non-parametric kernel-based drift field — no discriminator, no adversarial dynamics. The drift field provides a direct, geometry-aware signal: "move this sample toward that data point." This avoids mode collapse by construction (the repulsion term actively spreads generated samples), but requires large batches for the kernel computation to be effective.
