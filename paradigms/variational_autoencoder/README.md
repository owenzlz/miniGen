# VAE (Variational Autoencoder)

> Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014.

A VAE learns a latent representation $z$ of data $x$ by jointly training an **encoder** $q_\phi(z|x)$ and a **decoder** $p_\theta(x|z)$. Sampling is simple: draw $z$ from the prior and decode.

**Intuition**: A plain autoencoder can reconstruct data well, but its latent space is unstructured — sampling random codes produces garbage. The VAE fixes this by making the encoder output a *distribution* rather than a single point, and adding a KL penalty that pushes these distributions toward a standard Gaussian prior. This creates a smooth, continuous latent space where nearby codes decode to similar outputs, and random samples from $\mathcal{N}(0, \mathbf{I})$ land in meaningful regions. The tradeoff is that reconstruction becomes slightly blurry — the KL term prevents the encoder from memorizing exact codes.

---

## Model Architecture

The encoder maps data to a distribution in latent space; the decoder reconstructs from latent codes.

```
Encoder: x ∈ R² → MLP → (μ, log σ²) ∈ R^d × R^d
                              ↓ reparameterize
                         z = μ + σ ⊙ ε,  ε ~ N(0, I)
                              ↓
Decoder: z ∈ R^d → MLP → tanh → x̂ ∈ [-1, 1]²
```

```python
# models/vae_mlp.py: VAEModel
def encode(self, x):
    h = self.encoder_body(x)
    return self.fc_mu(h), self.fc_log_var(h)       # (μ, log σ²)

def decode(self, z):
    return self.decoder(z)                          # x̂ ∈ [-1, 1]
```

---

## Reparameterization Trick

To backpropagate through the stochastic sampling $z \sim q_\phi(z|x)$, we reparameterize:

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

where $\sigma = \exp(\tfrac{1}{2} \log \sigma^2)$. This moves the randomness into $\epsilon$, making $z$ a deterministic function of $(\mu, \log\sigma^2, \epsilon)$.

```python
# models/vae_mlp.py: reparameterize()
def reparameterize(self, mu, log_var):
    if self.training:
        std = torch.exp(0.5 * log_var)   # σ = exp(½ log σ²)
        eps = torch.randn_like(std)       # ε ~ N(0, I)
        return mu + eps * std             # z = μ + σ ⊙ ε
    return mu                             # At eval: use mean (no noise)
```

---

## Training Loss (ELBO)

The VAE maximizes the **Evidence Lower Bound (ELBO)**, or equivalently minimizes:

$$\mathcal{L} = \underbrace{-\mathbb{E}_{q_\phi(z|x)}\bigl[\log p_\theta(x|z)\bigr]}_{\text{Reconstruction loss}} + \underbrace{\beta \cdot D_\text{KL}\bigl(q_\phi(z|x)\;\|\;p(z)\bigr)}_{\text{KL divergence}}$$

### Reconstruction Loss

Measures how well the decoder reconstructs the input. Using Gaussian decoder → MSE:

$$\mathcal{L}_\text{recon} = \|x - \hat x\|^2$$

```python
# vae.py: training_loss()
x_recon, mu, log_var = model(x0)
recon_loss = F.mse_loss(x_recon, x0)
```

### KL Divergence

Regularizes the encoder to stay close to the prior $p(z) = \mathcal{N}(0, \mathbf{I})$. For Gaussian $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 I)$:

$$D_\text{KL}\bigl(\mathcal{N}(\mu, \sigma^2) \;\|\; \mathcal{N}(0, I)\bigr) = -\frac{1}{2}\sum_{j=1}^{d}\bigl(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\bigr)$$

```python
# vae.py: training_loss()
kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
```

### Combined Loss

$$\mathcal{L} = \mathcal{L}_\text{recon} + \beta \cdot \mathcal{L}_\text{KL}$$

$\beta = 1$ gives the standard ELBO. $\beta > 1$ is the **$\beta$-VAE** (Higgins et al., 2017), which encourages more disentangled latent representations at the cost of reconstruction quality.

```python
# vae.py: training_loss()
return recon_loss + self.beta * kl_loss
```

---

## Sampling

Generation is a single forward pass through the decoder:

$$z \sim \mathcal{N}(0, \mathbf{I}), \quad x = \text{Decoder}(z)$$

```python
# vae.py: sample()
z = torch.randn(num_samples, self.latent_dim, device=device)
return model.decode(z)
```

No iterative denoising — just one decoder call. This makes VAE sampling **much faster** than diffusion/flow models, but typically at the cost of sample quality.
