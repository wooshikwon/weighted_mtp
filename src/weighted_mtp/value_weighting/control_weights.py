"""Control weight generators for ablation experiments.

In the token-level advantage weighting study, the Critic model produces per-token
weights that guide LLM training. To verify that the Critic's signal is genuinely
informative (rather than an artifact of any non-uniform weighting), we need control
baselines that apply weights without real positional information.

Two controls are provided:

1. **Random Matched** -- Draws weights from a LogNormal distribution whose mean and
   standard deviation match the empirical Critic weight distribution. This tests
   whether the *shape* of the weight distribution alone (positive skew, heavy tail)
   is enough to explain training gains, independent of which tokens receive high
   weight.

2. **Shuffle Within Sequence** -- Takes the actual Critic weights and randomly
   permutes them across valid positions within each sequence. The per-sequence
   marginal distribution is perfectly preserved, but the assignment of weight to
   position is destroyed. This isolates the contribution of *positional*
   information from the Critic.

If the Critic's signal is informative, both controls should underperform the real
Critic weights while potentially outperforming uniform weighting (since they still
introduce variance that may act as a mild regularizer).
"""

import math

import torch


def generate_random_matched_weights(
    shape: tuple[int, ...],
    device: torch.device,
    target_mean: float = 1.328,
    target_std: float = 0.981,
    clip_min: float = 0.1,
    clip_max: float = 3.0,
) -> torch.Tensor:
    """Generate random weights whose distribution matches empirical Critic weights.

    The Critic's per-token weights are produced by an exp(A/beta) transform of
    advantage estimates, which yields a positively-skewed distribution well
    approximated by a LogNormal. This function samples from a LogNormal whose
    first two moments (mean, std) match the observed Critic statistics, then
    clamps to the same range used in production weighting.

    Args:
        shape: Desired tensor shape, typically ``(batch_size, seq_len)``.
        device: Torch device for the output tensor.
        target_mean: Mean of the empirical Critic weight distribution.
        target_std: Standard deviation of the empirical Critic weight distribution.
        clip_min: Lower clamp bound (mirrors production clipping).
        clip_max: Upper clamp bound (mirrors production clipping).

    Returns:
        A tensor of shape ``shape`` on ``device`` with values sampled from
        LogNormal(mu, sigma) and clamped to [clip_min, clip_max].
    """
    # Derive LogNormal parameters from the target mean and standard deviation.
    # For X ~ LogNormal(mu, sigma):
    #   E[X]   = exp(mu + sigma^2 / 2)
    #   Var[X] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
    # Solving for mu and sigma given E[X] and Std[X]:
    target_var = target_std**2
    sigma_sq = math.log(1 + target_var / target_mean**2)
    mu = math.log(target_mean) - sigma_sq / 2
    sigma = math.sqrt(sigma_sq)

    # Sample from LogNormal(mu, sigma) via the exp-of-Normal reparameterisation.
    normal_samples = torch.randn(shape, device=device)
    weights = torch.exp(mu + sigma * normal_samples)

    return weights.clamp(min=clip_min, max=clip_max)


def shuffle_weights_within_sequence(
    weights: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Shuffle Critic weights across valid positions within each sequence.

    This produces a control that has the *exact same* per-sequence weight
    distribution (mean, variance, min, max, histogram) as the real Critic
    weights, but with the positional assignment randomised. Padding positions
    (where ``attention_mask == 0``) are left unchanged.

    Args:
        weights: Critic-computed per-token weights, shape ``(batch_size, seq_len)``.
        attention_mask: Binary mask, shape ``(batch_size, seq_len)``. Ones indicate
            valid tokens; zeros indicate padding.

    Returns:
        A new tensor of the same shape with valid-position weights shuffled
        independently per sequence. The input tensor is not modified.
    """
    shuffled = weights.clone()
    batch_size = weights.size(0)

    for i in range(batch_size):
        valid_indices = attention_mask[i].nonzero(as_tuple=True)[0]
        if valid_indices.numel() <= 1:
            continue
        # Random permutation of the valid-position weights.
        perm = torch.randperm(valid_indices.numel(), device=weights.device)
        shuffled[i, valid_indices] = weights[i, valid_indices[perm]]

    return shuffled
