# AE (Autoencoder)

> Hinton & Salakhutdinov, "Reducing the Dimensionality of Data with Neural Networks", Science 2006.

An autoencoder learns to compress data into a low-dimensional latent code and reconstruct it back. The encoder maps $x \to z$, the decoder maps $z \to \hat{x}$, and training minimizes the reconstruction error $\|x - \hat{x}\|^2$.

**Intuition**: Unlike a VAE, the autoencoder has no regularization on its latent space — there is no KL term pushing $z$ toward a known prior like $\mathcal{N}(0, \mathbf{I})$. This means reconstruction can be near-perfect, but the latent space has no structure.

---

## Model Architecture

The encoder and decoder are symmetric MLPs connected through a bottleneck latent code.

```
Encoder: x ∈ R² → MLP(2→256, ReLU, [256→256, ReLU] × 2) → Linear(256→d) → z ∈ R^d
                                                                                ↓
Decoder: z ∈ R^d → MLP(d→256, ReLU, [256→256, ReLU] × 2) → Linear(256→2) → tanh → x̂ ∈ [-1, 1]²
```

```python
# ae_mlp.py: AEModel
def encode(self, x):
    h = self.encoder_body(x)     # MLP layers with ReLU
    return self.fc_latent(h)     # z ∈ R^d (single linear head, no mu/log_var split)

def decode(self, z):
    return self.decoder(z)       # x̂ ∈ [-1, 1] via tanh
```

Key difference from VAE: the encoder outputs a **single deterministic code** $z$ (not a distribution $(\mu, \log\sigma^2)$). There is no reparameterization trick.

---

## Training Loss

Pure reconstruction loss — minimize the MSE between input and reconstruction:

$$\mathcal{L} = \|x - \hat{x}\|^2 = \|x - \text{decode}(\text{encode}(x))\|^2$$

```python
# ae.py: training_loss()
x_recon = model(x0)           # encode → decode
return F.mse_loss(x_recon, x0)
```

No KL divergence, no regularization — the only objective is faithful reconstruction. This allows the encoder to use the latent space however it wants (arbitrary clustering, gaps, non-smooth mappings).

---

## Sampling

Since the latent space is unregularized, the AE **cannot generate novel samples** by drawing from a prior. Instead, it reconstructs existing data points:

```python
# ae.py: sample()
# Primary mode: reconstruct real data
x = collect_from_dataloader(num_samples)
return model(x)                            # encode → decode

# Fallback: decode from Gaussian noise (produces poor results)
z = torch.randn(num_samples, latent_dim, device=device)
return model.decode(z)
```

The primary sampling mode fetches real data from the dataloader, encodes it, and decodes it — this tests reconstruction quality. The Gaussian fallback illustrates the failure mode: without regularization, decoding random $z$ vectors produces incoherent outputs.



