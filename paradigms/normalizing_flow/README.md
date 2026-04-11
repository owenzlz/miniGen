# Normalizing Flow (RealNVP)

> Dinh et al., "Density estimation using Real-NVP", ICLR 2017.

A Normalizing Flow learns an **invertible** mapping $f: x \to z$ between data space and a simple latent distribution (standard normal). Because $f$ is bijective with a tractable Jacobian, we can compute **exact likelihoods** — no ELBO bound or adversarial training needed.

**Intuition**: Imagine stretching and folding a sheet of rubber with ink dots arranged in a perfect Gaussian pattern. Each simple stretch warps the dots a little, and by composing many such stretches you can rearrange them into any complex pattern — spirals, clusters, etc. A normalizing flow does exactly this: it learns a chain of simple invertible transformations that warp a Gaussian into the data distribution. Because every transformation is invertible, we can both generate samples (Gaussian → data via the inverse) and compute exact likelihoods (data → Gaussian via the forward pass). The tradeoff is architectural: each transformation must be invertible with a cheap Jacobian determinant, which constrains the design — hence the coupling layer trick where half the dimensions pass through unchanged while the other half are transformed.

---

## Model Architecture

RealNVP stacks affine coupling layers with **alternating masks** — each layer transforms one dimension conditioned on the other. The forward pass (data → latent) is used for training; the inverse (latent → data) is used for sampling.

```
Forward (training):   x ∈ R²  →  Coupling(mask=0) → Coupling(mask=1) → ... × K  →  z ∈ R²
                                  each layer accumulates log|det J|

Inverse (sampling):   z ~ N(0,I)  →  Coupling_K⁻¹ → ... → Coupling₂⁻¹ → Coupling₁⁻¹  →  x ∈ R²
```

Each coupling layer's internal network: `x_fixed ∈ R¹ → MLP(1→128) + ReLU × 2 → Linear(128→2) → (s, t)`

```python
# realnvp.py: RealNVP2D
self.layers = nn.ModuleList([
    AffineCouplingLayer(mask_idx=i % 2, hidden_size=128, depth=2)
    for i in range(num_coupling_layers)          # default: 8 layers
])

def forward(self, x):                            # x → z (training)
    log_det_total = zeros(B)
    for layer in self.layers:
        x, log_det = layer(x)
        log_det_total += log_det
    return x, log_det_total

def inverse(self, z):                            # z → x (sampling)
    for layer in reversed(self.layers):
        z = layer.inverse(z)
    return z
```

---

## Change of Variables

For an invertible transformation $z = f(x)$ with $x = f^{-1}(z)$:

$$\log p(x) = \log p_z\bigl(f(x)\bigr) + \log\bigl|\det J_f(x)\bigr|$$

where $J_f(x) = \frac{\partial f}{\partial x}$ is the Jacobian. This is exact — no approximation.

---

## Affine Coupling Layer

RealNVP uses **affine coupling layers** that split the input into two parts: one stays fixed, the other is transformed conditioned on the fixed part.

For 2D data $x = (x_\text{fixed}, x_\text{change})$:

**Forward** (data → latent):

$$y_\text{fixed} = x_\text{fixed}$$

$$y_\text{change} = x_\text{change} \cdot \exp(s) + t, \quad \text{where } (s, t) = \text{NN}(x_\text{fixed})$$

**Inverse** (latent → data):

$$x_\text{change} = (y_\text{change} - t) \cdot \exp(-s)$$

The log-determinant is simply $s$ (the log-scale), because the Jacobian is triangular:

$$\log\bigl|\det J\bigr| = s$$

```python
# realnvp.py: AffineCouplingLayer
def forward(self, x):
    x_fixed = x[:, self.mask_idx : self.mask_idx + 1]
    x_change = x[:, self.change_idx]

    st = self.net(x_fixed)              # NN(x_fixed) → (s, t)
    s = torch.tanh(st[:, 0])            # Bounded scale for stability
    t = st[:, 1]

    y_change = x_change * torch.exp(s) + t
    y = x.clone()
    y[:, self.change_idx] = y_change
    return y, s                          # log_det = s

def inverse(self, y):
    y_fixed = y[:, self.mask_idx : self.mask_idx + 1]
    y_change = y[:, self.change_idx]

    st = self.net(y_fixed)
    s = torch.tanh(st[:, 0])
    t = st[:, 1]

    x_change = (y_change - t) * torch.exp(-s)  # Invert the affine transform
    x = y.clone()
    x[:, self.change_idx] = x_change
    return x
```

### Stacking Layers

Multiple coupling layers are stacked with **alternating masks** (which dimension is fixed vs. transformed). This ensures both dimensions get transformed:

```python
# realnvp.py: RealNVP2D
self.layers = nn.ModuleList([
    AffineCouplingLayer(mask_idx=i % 2, ...)    # Alternates: 0, 1, 0, 1, ...
    for i in range(num_coupling_layers)
])
```

The total log-determinant is the sum over all layers:

```python
# realnvp.py: RealNVP2D.forward()
log_det_total = torch.zeros(x.shape[0], device=x.device)
for layer in self.layers:
    x, log_det = layer(x)
    log_det_total = log_det_total + log_det       # Sum log-dets
return x, log_det_total
```

---

## Training (Maximum Likelihood)

Minimize the **negative log-likelihood**:

$$\mathcal{L} = -\mathbb{E}_{x \sim p_\text{data}}\bigl[\log p_z(f(x)) + \log|\det J_f(x)|\bigr]$$

The log-prior under standard normal is:

$$\log p_z(z) = -\frac{1}{2}\sum_j \bigl(z_j^2 + \log 2\pi\bigr)$$

```python
# normalizing_flow.py: training_loss()
z, log_det = model(x0)                          # Forward: x → z, log|det J|

# Log-prior: log N(z; 0, I)
log_prior = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1)

# log p(x) = log p(z) + log |det J|
log_likelihood = log_prior + log_det

return -log_likelihood.mean()                    # NLL loss
```

---

## Sampling

Sampling inverts the flow — draw from the prior, then pass through the inverse:

$$z \sim \mathcal{N}(0, \mathbf{I}), \quad x = f^{-1}(z)$$

The inverse applies coupling layers in **reverse order**:

```python
# normalizing_flow.py: sample()
z = torch.randn(num_samples, input_dim, device=device)
return model.inverse(z)
```

```python
# realnvp.py: RealNVP2D.inverse()
for layer in reversed(self.layers):    # Reverse order!
    z = layer.inverse(z)
return z
```
