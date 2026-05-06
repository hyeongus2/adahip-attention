# AdaHiP Attention (Adaptive Hierarchically Pruned Attention)

> This repository contains an experimental extension of HiP Attention,
> implementing AdaHiP proposed in my research.

## Overview

AdaHiP (Adaptive Hierarchically Pruned Attention) improves HiP by dynamically adjusting
the retrieval budget k based on attention entropy.

While HiP uses a fixed k for all queries, AdaHiP allocates computation adaptively:

- Low-entropy queries → small k (reduce compute)
- High-entropy queries → large k (preserve accuracy)

This enables a better trade-off between perplexity and speed in long-context inference.

## Key Idea

AdaHiP estimates normalized attention entropy H(q) and computes:

k_dyn = RoundUpTo64(k_min + (k_max - k_min) * x)

where:
- x = (alpha * H(q))^p
- k_min = max(64, k_base / 4)
- k_max = 2 * k_base

## Why AdaHiP?

HiP suffers from a static sparsity problem:

- Fixed k is too large for simple queries → wasted compute
- Fixed k is too small for complex queries → important tokens are missed

AdaHiP solves this by adapting k based on query complexity.

## Experimental Results

| Method | PPL ↓ | Speedup ↑ |
|--------|------|----------|
| FA2    | 12.30 | 1.00x |
| HiP    | 12.63 | 2.18x |
| AdaHiP | 12.46 | 2.08x |

AdaHiP improves perplexity compared to static HiP,
while maintaining sub-quadratic efficiency.

## Getting Started

Follow the original HiP setup instructions:

```bash
git clone https://github.com/hyeongus2/adahip-attention.git
cd adahip-attention
