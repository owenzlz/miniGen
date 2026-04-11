# GAN (Generative Adversarial Network)

> Goodfellow et al., "Generative Adversarial Nets", NeurIPS 2014.

A GAN trains two networks in a **minimax game**: a **Generator** $G$ that produces fake samples, and a **Discriminator** $D$ that distinguishes real from fake. At equilibrium, the generator produces samples indistinguishable from real data.

**Intuition**: Think of a counterfeiter (generator) trying to produce fake currency and a detective (discriminator) trying to spot the fakes. The counterfeiter gets better by learning from the detective's feedback, while the detective improves by seeing more sophisticated counterfeits. This adversarial pressure drives both networks to improve until the generator's output is indistinguishable from real data. Unlike VAEs or diffusion models, GANs don't model an explicit density or reconstruction objective — the generator simply learns to fool the discriminator, which implicitly forces it to match the data distribution.

---

## Architecture

```
Generator:     z ~ N(0, I) ∈ R^d  →  MLP + ReLU  →  tanh  →  x_fake ∈ [-1, 1]²
Discriminator: x ∈ R²             →  MLP + LeakyReLU       →  logit ∈ R  (real/fake)
```

```python
# models/gan_mlp.py
class Generator(nn.Module):
    def forward(self, z, **kwargs):
        return self.net(z)           # z → x_fake ∈ [-1, 1]²

class Discriminator(nn.Module):
    def forward(self, x, **kwargs):
        return self.net(x)           # x → logit (apply sigmoid for probability)
```

---

## Training

GAN uses **alternating optimization** — the discriminator and generator are updated separately. This is why GAN uses `train_step()` instead of the standard `training_loss()` interface.

### Discriminator Update

The discriminator maximizes its ability to classify real vs fake:

$$\mathcal{L}_D = -\mathbb{E}_{x \sim p_\text{data}}\bigl[\log D(x)\bigr] - \mathbb{E}_{z \sim p_z}\bigl[\log\bigl(1 - D(G(z))\bigr)\bigr]$$

Using binary cross-entropy with logits:

```python
# gan.py: train_step() — Discriminator
z = torch.randn(batch_size, self.latent_dim, device=device)
with torch.no_grad():
    x_fake = generator(z)              # Generate fake samples (no grad for G)

d_real = self.discriminator(x_real)     # D(x_real) — should be high
d_fake = self.discriminator(x_fake)     # D(x_fake) — should be low

d_loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
d_loss = d_loss_real + d_loss_fake
```

The discriminator can be updated multiple times per generator step (`n_critic`):

```python
for _ in range(self.n_critic):
    # ... discriminator update above ...
    self.disc_optimizer.zero_grad()
    d_loss.backward()
    self.disc_optimizer.step()
```

### Generator Update (Non-Saturating Loss)

Instead of the original minimax objective $\min_G \mathbb{E}[\log(1 - D(G(z)))]$ which saturates when $D$ is strong, we use the **non-saturating** formulation:

$$\mathcal{L}_G = -\mathbb{E}_{z \sim p_z}\bigl[\log D(G(z))\bigr]$$

This provides stronger gradients early in training.

```python
# gan.py: train_step() — Generator
z = torch.randn(batch_size, self.latent_dim, device=device)
x_fake = generator(z)
d_fake = self.discriminator(x_fake)

# Non-saturating: -E[log D(G(z))]
# Implemented as BCE with target=1 (trick the discriminator)
g_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))

gen_optimizer.zero_grad()
g_loss.backward()
nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
gen_optimizer.step()
```

---

## Sampling

Like the VAE, sampling is a single forward pass — no iterative process:

$$z \sim \mathcal{N}(0, \mathbf{I}), \quad x = G(z)$$

```python
# gan.py: sample()
z = torch.randn(num_samples, self.latent_dim, device=device)
return model(z)
```

---

## Integration with train.py

GAN is special-cased in the training loop because it needs alternating updates:

```python
# train.py (simplified)
if hasattr(process, 'train_step'):
    # GAN path: process manages both G and D updates
    loss_val = process.train_step(model, x, optimizer, y=y, grad_clip=grad_clip)
else:
    # Standard path: process.training_loss() + optimizer.step()
    loss = process.training_loss(model, x, t, y=y)
```
