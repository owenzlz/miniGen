# AR (Autoregressive Generation)

> The autoregressive factorization $p(x) = \prod_d p(x_d \mid x_1, \dots, x_{d-1})$ is the foundation of GPT, PixelCNN, and modern AR image generators.

An autoregressive model decomposes the joint distribution into a product of conditionals — each dimension is predicted one at a time, conditioned on all previously generated dimensions. Coordinates are discretized into bins and the model predicts a categorical distribution over bins for each dimension.

**Intuition**: Instead of generating all dimensions of a data point at once, AR models generate one dimension at a time in a fixed order: first predict $x_1$, then predict $x_2$ given $x_1$, and so on. This is exactly how language models work — predicting the next token given all previous tokens. For our 2D points, the model factorizes $p(x_1, x_2) = p(x_1) \cdot p(x_2 \mid x_1)$: first choose the x-coordinate, then choose the y-coordinate conditioned on x. Continuous coordinates are discretized into bins, turning regression into classification. This approach provides exact likelihood computation and stable training (just cross-entropy), but sequential generation means sampling scales linearly with dimensionality.

---

## Model Architecture

A tiny GPT-style causal transformer treats each coordinate as a token in a length-2 sequence.

```
Input:  [START,  embed(x₁)]     ← shifted: each position sees only previous tokens
           ↓         ↓
       Pos Embed + (optional) Class Embed
           ↓         ↓
       [Causal Transformer × 2]
           ↓         ↓
Output: [logits₁, logits₂]     ← categorical distribution over 128 bins per dimension
```

```python
# ar_transformer.py: ARTransformer2D.forward()
# Build shifted input: [START, embed(x_1), ..., embed(x_{D-1})]
start = self.start_token.expand(B, 1, -1)       # learnable start token
tok = self.token_embed(x_bins[:, :-1])           # embed all but last dim
h = cat([start, tok], dim=1)                     # (B, D, H)

h = h + self.pos_embed(positions)                # positional embedding
for block in self.blocks:                        # causal transformer
    h = block(h)
logits = self.head(self.ln_f(h))                 # (B, D, num_bins)
```

**Causal masking** ensures that the prediction for dimension $d$ only depends on dimensions $x_1, \dots, x_{d-1}$. Each transformer block is pre-norm with multi-head self-attention and a GELU FFN.

---

## Discretization

Continuous coordinates in $[-1, 1]$ are mapped to discrete bins in $[0, N-1]$ where $N$ is the number of bins:

$$\text{discretize}(x) = \text{round}\!\left(\frac{x + 1}{2} \cdot (N - 1)\right)$$

$$\text{undiscretize}(b) = \frac{b}{N - 1} \cdot 2 - 1$$

```python
# autoregressive.py
def discretize(self, x):
    return ((x.clamp(-1, 1) + 1) / 2 * (self.num_bins - 1)).round().long()

def undiscretize(self, bins):
    return bins.float() / (self.num_bins - 1) * 2 - 1
```

With `num_bins=128`, the resolution is $\approx 0.016$ per bin over $[-1, 1]$.

---

## Training: Teacher Forcing with Cross-Entropy

During training, the model receives the ground-truth bins at all positions (teacher forcing) and predicts the categorical distribution for each dimension:

$$\mathcal{L} = -\sum_{d=1}^{D} \log p_\theta(x_d \mid x_1, \dots, x_{d-1})$$

This is simply **cross-entropy** between the predicted logits and the true bin indices.

```python
# autoregressive.py: training_loss()
x_bins = self.discretize(x0)                    # (B, D) continuous → discrete
logits = model(x_bins, y=y)                      # (B, D, num_bins)
loss = F.cross_entropy(
    logits.reshape(-1, self.num_bins),           # flatten to (B*D, num_bins)
    x_bins.reshape(-1),                          # flatten to (B*D,)
)
```

All dimensions are predicted in a single forward pass (parallelized via causal masking), even though generation is sequential.

---

## Sampling: Sequential Generation

Samples are generated dimension by dimension — each step samples from the predicted categorical distribution, then feeds the result back as input for the next dimension:

```python
# autoregressive.py: sample()
x_bins = torch.zeros(B, D, device=device, dtype=torch.long)

for d in range(D):
    logits = model(x_bins, y=y)                  # (B, D, num_bins)
    probs = softmax(logits[:, d] / temperature)  # categorical over bins for dim d
    x_bins[:, d] = multinomial(probs, 1)         # sample one bin

return self.undiscretize(x_bins)                 # back to [-1, 1]
```

**Temperature** controls sharpness: $T < 1$ concentrates probability on the most likely bins (sharper, less diverse), $T > 1$ spreads probability more evenly (softer, more diverse), $T = 1$ is the learned distribution.

For 2D, this is just 2 sequential steps. 