# EBM (Energy-Based Model)

> Du & Mordatch, "Implicit Generation and Modeling with Energy-Based Models", NeurIPS 2019.

An EBM defines the data distribution as $p(x) \propto \exp(-E_\theta(x))$, where $E_\theta(x)$ is a learned scalar energy function. Low energy = high probability. Training uses **Contrastive Divergence** to shape the energy landscape; sampling uses **Langevin MCMC** to draw from it.

**Intuition**: Think of the energy function as a terrain. Data points sit in valleys (low energy) and empty regions are hills (high energy). Training carves this terrain by digging valleys where real data lives and raising hills where the model's own samples (negatives) land. Over time, the landscape converges so that its valleys match the true data distribution. To generate new samples, we drop a ball at a random location and let it roll downhill (Langevin dynamics) — it naturally settles into a valley, producing a sample from the learned distribution.

Unlike models that learn to directly map noise to data (VAE, GAN, flow), an EBM only learns a scalar **"how good does this point look?"** score for any location in space. This is powerful — the model doesn't need to define an explicit mapping or a normalized density — but it comes at a cost: both training and sampling require iterative MCMC, which is slower and harder to tune than a single forward pass.

---

## Model Architecture

The energy network maps each input point to a scalar energy value.

```
EnergyMLP: x ∈ R² → Linear(2→256) → [ResBlock × 6] → Linear(256→1) → E(x) ∈ R
```

Each **ResBlock** = `Linear → SiLU → Linear` with a residual connection:

```python
# energy_mlp.py: EnergyMLP.forward()
h = self.input_proj(x)
for i in range(self.depth):
    residual = h
    h = Linear(h) → SiLU → Linear(h)
    h = h + residual                      # residual connection
energy = self.output_proj(h).squeeze(-1)  # (B,) scalar per sample
```

No time or class conditioning — EBMs model a static energy landscape.

Optional **spectral normalization** on all linear layers constrains the Lipschitz constant. Disabled in best config (too flat for fine structure like spiral arms).

---

## Training: Contrastive Divergence

The loss pushes energy **down** on real data and **up** on model-generated negatives:

$$\mathcal{L} = \underbrace{\mathbb{E}_{p_\text{data}}[E(x)]}_{\text{lower real energy}} - \underbrace{\mathbb{E}_{p_\text{neg}}[E(x^-)]}_{\text{raise negative energy}} + \underbrace{\lambda \cdot \mathbb{E}[(E^2)]}_{\text{regularization}}$$

### Step 1: Energy on Real Data

```python
# ebm.py: training_loss()
energy_real = model(x0)
```

### Step 2: Generate Negatives via Langevin MCMC

Negatives are initialized and then refined by short-run Langevin MCMC to produce samples near the model's current low-energy regions.

**Initialization**: With `cd_data_init=true`, half start from **data + noise** (local structure) and half from a **replay buffer** (global diversity). The replay buffer is a circular store of 10k recent negatives, with 5% fresh noise injection on each draw (`reinit_freq=0.05`).

**MCMC refinement**: Run `cd_steps=60` Langevin steps on the initialized negatives:

$$x^- \leftarrow x^- - \frac{\alpha}{2} \nabla_{x^-} E(x^-) + \eta \cdot \sqrt{\alpha} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

With `cd_noise_scale=0.1`, the update is **gradient-dominated** — MCMC focuses on descending the energy gradient rather than random exploration, producing sharper negatives.

```python
# ebm.py: training_loss()
# Initialize negatives
data_init = x0[:n_data] + noise_std * randn(...)   # 50% local (data + noise)
buffer_init = sample_from_buffer(n_buffer, ...)     # 50% global (replay buffer)
x_neg_init = cat([data_init, buffer_init])

# Refine via short-run Langevin MCMC
x_neg = langevin_mcmc(model, x_neg_init, steps=60, step_size=0.1, noise_scale=0.1)
```

```python
# ebm.py: _langevin_mcmc()
for i in range(steps):
    energy = model(x).sum()
    grad_x = torch.autograd.grad(energy, x)[0]
    x = x - (step_size / 2) * grad_x + noise_scale * sqrt(step_size) * randn_like(x)
```

Per-sample **gradient clipping** (`langevin_grad_clip=1.0`) prevents explosive updates without limiting model expressivity (replaces spectral norm for stability).

### Step 3: Compute Loss

Negatives are **detached** — no backprop through the MCMC chain:

```python
# ebm.py: training_loss()
energy_neg = model(x_neg.detach())

cd_loss = energy_real.mean() - energy_neg.mean()
reg_loss = energy_reg * (energy_real ** 2 + energy_neg ** 2).mean()
return cd_loss + reg_loss
```

The regularization term (`energy_reg=0.1`) prevents energy magnitudes from diverging.

---

## Sampling: Annealed Langevin MCMC

Generation starts from random noise and iteratively moves toward low-energy regions:

$$x_0 \sim \mathcal{N}(0, \mathbf{I}), \quad x_{i+1} = x_i - \frac{\alpha}{2} \nabla_x E(x_i) + \sigma_i \sqrt{\alpha} \cdot \epsilon$$

```python
# ebm.py: sample()
x = torch.randn(shape, device=device)
with torch.enable_grad():       # needed even under outer no_grad()
    x = _langevin_mcmc(model, x, steps=3000, step_size=0.1,
                       noise_scale_mult=0.5, noise_end_mult=0.01)
```

### Annealed Noise Schedule

The noise multiplier $\sigma_i$ linearly decays from `0.5` to `0.01` over 3000 steps:

- **Early steps** (high noise): broad exploration to find the general region of modes.
- **Late steps** (low noise): fine-grained exploitation to converge onto precise mode locations.

This is critical — constant noise either fails to find modes (too low) or fails to converge (too high).

### Stability Controls

- **Gradient clipping** (`langevin_grad_clip=1.0`): caps per-sample gradient norm to prevent explosive jumps.
- **Coordinate clamping** (`langevin_clamp=5.0`): prevents samples from drifting to infinity.
- **`torch.enable_grad()`**: re-enables autograd locally since `sample()` may be called under an outer `@torch.no_grad()` wrapper.

---

## Replay Buffer

The replay buffer stores recent negatives and provides warm-start points for MCMC chains:

- **Size**: 10,000 samples, initialized to random noise.
- **Draw**: 95% from buffer + 5% fresh noise on each training step.
- **Update**: circular write — new negatives overwrite old ones sequentially.
- **Purpose**: prevents mode collapse by maintaining diverse negatives and avoiding cold-start MCMC every step.

---

## Key Design Decisions

EBMs were quite hard to train in my experiments. We found the following settings for the best possible results. However, please note that this might not be optimal. 

| Decision | Rationale |
|---|---|
| No spectral norm | SN constrains the Lipschitz constant, making the energy landscape too smooth to distinguish close spiral arms (~0.5 apart) |
| Gradient clip instead | Provides MCMC stability without limiting model expressivity |
| Annealed inference noise | Random init can't find modes with constant low noise; constant high noise prevents convergence |
| Data-initialized CD | Local negatives (data+noise) teach fine structure; buffer negatives teach global landscape |
| Low training noise (0.1) | Gradient-dominated MCMC produces negatives closer to learned modes |
| Energy regularization (0.1) | Prevents the energy gap between real and fake from growing unboundedly |
