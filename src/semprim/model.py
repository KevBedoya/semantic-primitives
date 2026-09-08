"""The simplex bottleneck and its decoder.

    h(t) = frozen encoder output              (cached, R^d_enc)
    z(t) = W2 GELU(W1 h + b1) + b2            (R^N)
    a(t) = entmax(z / tau)                    (simplex)
    y^(t) = D a(t)                            (R^d_teach)

WHY Q IS NOT HERE
-----------------
The design brief writes the embedding as f(t) = Q a(t) with Q orthonormal. That
matrix is deliberately absent, because the brief's own algebra removes it: since
Q^T Q = I,

    f(t)^T f(s) = a(t)^T Q^T Q a(s) = a(t)^T a(s),

so under inner products or Euclidean distances the orientation of Q adds no
expressive power whatsoever -- every kernel in kernels.py would return the same
number. Worse, without an external grounding objective Q is not identifiable, so
training one is training a rotation nothing constrains. Q = I is therefore not a
simplification with a cost; it is the same model with a redundant parameter
removed. ``embedding()`` below exists to make the equivalence explicit for
anything downstream that expects f(t).

If a later milestone genuinely needs a rotated basis -- only grounding
supervision could make it identifiable -- the exact parameterisation is
Q = expm(A - A^T) for an unconstrained square A, applied in ``embedding()``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reference implementation from the paper's authors (Peters, Martins & Vieira,
# ACL 2019). Imported lazily-ish so that importing this module for its shapes
# does not hard-fail on a machine without it; the error surfaces at construction
# with an actionable message instead of an ImportError at import time.
try:
    from entmax import entmax_bisect, entmax15, sparsemax
    _HAVE_ENTMAX = True
except ImportError:  # pragma: no cover
    _HAVE_ENTMAX = False


def simplex_project(z: torch.Tensor, alpha: float, tau: torch.Tensor) -> torch.Tensor:
    """Map scores onto the simplex with the requested entmax family member.

    alpha = 1 is softmax (dense: every primitive gets positive mass).
    alpha = 2 is sparsemax (exact zeros; Martins & Astudillo, 2016).
    In between, alpha-entmax interpolates -- sparse but with a smoother
    transition than sparsemax.

    WHY NOT JUST USE SPARSEMAX
    --------------------------
    Sparsemax is what the brief asks for, and exact zeros are the goal. But
    sparsemax has *zero gradient* for coordinates outside its support: once a
    primitive drops out early in training, no gradient can ever bring it back.
    On a bottleneck whose entire purpose is a reusable inventory of N primitives
    that is a dead-unit trap, and it is silent -- the model trains fine, it just
    quietly uses fewer primitives than you think. Annealing alpha from 1.0
    upward keeps every coordinate alive while the representation forms, then
    tightens. train.py drives the schedule; diagnostics.dead_coordinates()
    measures whether it worked.
    """
    if not _HAVE_ENTMAX and alpha != 1.0:
        raise ImportError(
            "alpha > 1 needs the 'entmax' package (pip install entmax). "
            "alpha = 1.0 (softmax) works without it.")
    scaled = z / tau
    if alpha == 1.0:
        return F.softmax(scaled, dim=-1)
    if alpha == 2.0:
        return sparsemax(scaled, dim=-1)
    if alpha == 1.5:
        return entmax15(scaled, dim=-1)          # closed form, faster than bisect
    return entmax_bisect(scaled, alpha=alpha, dim=-1)


class SimplexBottleneck(nn.Module):
    """Encoder output -> mixture coefficients over N primitives.

    ``tau`` is parameterised in log space so it stays positive under
    unconstrained optimisation. It is learnable by default because it, not the
    head weights, is what actually controls how peaked the mixtures are: fixing
    it to 1.0 makes sparsity depend on the arbitrary scale of z.
    """

    def __init__(self, in_dim: int = 768, hidden: int = 1024,
                 n_primitives: int = 256, alpha: float = 1.0,
                 tau_init: float = 1.0, learn_tau: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.n_primitives = n_primitives
        self.alpha = float(alpha)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, n_primitives),
        )
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log(),
                                    requires_grad=learn_tau)

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def set_alpha(self, alpha: float) -> None:
        """Retune sparsity mid-training. Called by the annealing schedule."""
        self.alpha = float(alpha)

    def scores(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return simplex_project(self.scores(h), self.alpha, self.tau)


class PrimitiveModel(nn.Module):
    """Bottleneck + linear decoder back into teacher space.

    The decoder is deliberately linear and bias-free, so that

        y^(t) = D a(t),   a(t) in the simplex

    places every reconstruction in the *convex hull* of the columns of D. Each
    column d_i is then exactly the teacher-space interpretation of primitive i,
    which is what makes coordinates inspectable at all -- and it means this
    model is amortised archetypal analysis (Cutler & Breiman, 1994). See
    archetypal.py for the non-amortised fit that bounds what this can achieve.

    Adding a bias or a nonlinearity here would buy reconstruction accuracy and
    destroy that reading, so neither is an option.
    """

    def __init__(self, in_dim: int = 768, teacher_dim: int = 768,
                 hidden: int = 1024, n_primitives: int = 256,
                 alpha: float = 1.0, tau_init: float = 1.0,
                 learn_tau: bool = True, dropout: float = 0.0):
        super().__init__()
        self.bottleneck = SimplexBottleneck(
            in_dim=in_dim, hidden=hidden, n_primitives=n_primitives,
            alpha=alpha, tau_init=tau_init, learn_tau=learn_tau,
            dropout=dropout)
        self.decoder = nn.Linear(n_primitives, teacher_dim, bias=False)
        self.n_primitives = n_primitives

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(h)

    def reconstruct(self, alpha: torch.Tensor) -> torch.Tensor:
        return self.decoder(alpha)

    def embedding(self, alpha: torch.Tensor) -> torch.Tensor:
        """f(t) = Q a(t) with Q = I -- see the module docstring.

        Present so callers expecting the brief's f(t) have something to call,
        and so the one place that would need changing if Q ever becomes real is
        obvious. Do NOT L2-normalise the result: that breaks the convex
        combination. Normalise a kernel feature map instead.
        """
        return alpha

    @torch.no_grad()
    def init_decoder_(self, archetypes: torch.Tensor) -> None:
        """Seed D from archetypes fitted on teacher embeddings (N, d_teach).

        Random initialisation makes the decoder start with N meaningless columns
        that the reconstruction loss must drag into position, which wastes the
        early epochs and risks settling into a poor local arrangement. Starting
        from an actual archetypal / NMF fit means column i already denotes
        something before the first gradient step.
        """
        expected = (self.n_primitives, self.decoder.out_features)
        if tuple(archetypes.shape) != expected:
            raise ValueError(
                f"expected archetypes of shape "
                f"({self.n_primitives}, {self.decoder.out_features}), "
                f"got {tuple(archetypes.shape)}")
        # nn.Linear stores weight as (out_features, in_features) = (d_teach, N),
        # so the columns of D are the rows of `archetypes` transposed.
        self.decoder.weight.copy_(archetypes.t().to(self.decoder.weight.dtype))
