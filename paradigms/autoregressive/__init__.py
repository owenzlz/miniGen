"""
Autoregressive generative process.

Factorizes p(x) = prod_d p(x_d | x_{<d}) using discretized coordinates
and a tiny causal transformer.
"""
from paradigms.autoregressive.autoregressive import AutoregressiveGeneration
from paradigms.autoregressive.ar_transformer import ARTransformer2D

__all__ = ["AutoregressiveGeneration", "ARTransformer2D"]
