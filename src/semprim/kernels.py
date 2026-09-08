"""Positive-semidefinite kernels over simplex coordinates.

Every kernel here exposes ``feature_map(a)`` satisfying, exactly,

    k(a, b) == feature_map(a) @ feature_map(b).T

That identity is not decoration. It is what makes these usable for retrieval: an
index stores ``feature_map(alpha)`` once and answers nearest-neighbour queries
with an ordinary inner product, so FAISS -- or anything else that only knows how
to do dot products -- needs no knowledge of which kernel was trained.
tests/test_kernels.py asserts it for all four to 1e-6.

The inputs are always points on the simplex: ``a >= 0`` and ``sum(a) == 1``. Two
consequences worth stating once, because both are easy to forget downstream:

  * ``a_i`` is a *share* of semantic mass, not the probability that primitive i
    is present. Raising one coordinate necessarily lowers others.
  * Do not L2-normalise the bottleneck output itself; that destroys the convex
    combination. Normalise a kernel feature map instead, which is what
    ``LearnedKernel(normalise=True)`` and ``MixtureKernel`` do.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _check_simplex(a: torch.Tensor, tol: float = 1e-4, name: str = "alpha") -> None:
    """Cheap debug assertion. Off the hot path -- callers opt in."""
    if (a < -tol).any():
        raise ValueError(f"{name} has negative entries (min {a.min().item():.3e})")
    sums = a.sum(dim=-1)
    if (sums - 1.0).abs().max() > tol:
        raise ValueError(
            f"{name} rows do not sum to 1 (max deviation "
            f"{(sums - 1.0).abs().max().item():.3e})")


class Kernel(nn.Module):
    """Base class: a PSD kernel with an explicit finite-dimensional feature map."""

    def feature_map(self, a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Pairwise kernel matrix between rows of ``a`` (n, N) and ``b`` (m, N)."""
        return self.feature_map(a) @ self.feature_map(b).transpose(-1, -2)

    def diagonal(self, a: torch.Tensor) -> torch.Tensor:
        """k(a_i, a_i) for each row, without forming the full matrix."""
        phi = self.feature_map(a)
        return (phi * phi).sum(dim=-1)


class LinearKernel(Kernel):
    """k(a, b) = a . b -- shared contribution to the same primitives.

    The strictest reading of the bottleneck: two texts are similar only insofar
    as they put mass on literally the same primitives. Distinct primitives are
    orthogonal, so related-but-different concepts score zero. That limitation is
    what LearnedKernel exists to lift.
    """

    def feature_map(self, a: torch.Tensor) -> torch.Tensor:
        return a


class HellingerKernel(Kernel):
    """k(a, b) = sum_i sqrt(a_i b_i) -- the Hellinger (Bhattacharyya) affinity.

    A member of the probability product kernel family (Jebara et al., 2004). PSD
    by construction with feature map sqrt(a), bounded in [0, 1], and k(a, a) = 1
    exactly because sum(a) = 1. Those three properties together make it the right
    default: scores are directly comparable across texts with no normalisation
    step to get wrong.

    Note sqrt has infinite gradient at 0, which bites once the bottleneck starts
    producing exact zeros (entmax with alpha > 1). ``eps`` keeps the backward
    pass finite; it perturbs the kernel value by at most N*eps.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def feature_map(self, a: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(a.clamp_min(self.eps))


class LearnedKernel(Kernel):
    """k(a, b) = a^T B b with B = R^T R + eps*I, so B is PSD by construction.

    This is the answer to the main cost of orthogonality: distinct primitive
    vertices have zero ordinary dot product even when their meanings are related.
    Parameterising B rather than constraining it after the fact means the model
    can *learn* that primitives i and j are related -- B_ij is exactly the kernel
    between the two primitive vertices -- while the coordinates themselves stay
    orthogonal in the original Euclidean inner product. The two requirements are
    compatible because they are statements about different inner products.

    The feature map is exact rather than approximate:

        a^T (R^T R + eps I) b = (Ra) . (Rb) + eps (a . b)

    so phi(a) = [Ra ; sqrt(eps) a], of dimension rank + N.

    ``eps > 0`` is not cosmetic: it makes B strictly positive definite, so
    k(a, a) > 0 for every a and the diagonal normalisation below cannot divide by
    zero even if R collapses to zero during training.
    """

    def __init__(self, n_primitives: int, rank: int | None = None,
                 eps: float = 1e-4, normalise: bool = True):
        super().__init__()
        self.n_primitives = n_primitives
        self.rank = rank or n_primitives
        self.eps = eps
        self.normalise = normalise
        # Small init: B starts near eps*I, i.e. near the linear kernel scaled
        # down. Starting near identity rather than at a random PSD matrix means
        # training departs from the interpretable baseline rather than from
        # noise, so any learned off-diagonal structure is attributable.
        self.R = nn.Parameter(torch.randn(self.rank, n_primitives) * 0.01)

    def matrix(self) -> torch.Tensor:
        """The full B. Only for inspection and tests -- training never needs it."""
        eye = torch.eye(self.n_primitives, device=self.R.device, dtype=self.R.dtype)
        return self.R.transpose(0, 1) @ self.R + self.eps * eye

    def feature_map(self, a: torch.Tensor) -> torch.Tensor:
        phi = torch.cat([a @ self.R.transpose(0, 1), (self.eps ** 0.5) * a], dim=-1)
        if self.normalise:
            # Diagonal normalisation, i.e. k~(a,b) = k(a,b)/sqrt(k(a,a)k(b,b)).
            # Applied to the feature map so the identity k = phi.phi still holds
            # and the result stays usable in a dot-product index.
            phi = phi / phi.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return phi


class MixtureKernel(Kernel):
    """k = (1 - lam) * k_H + lam * k~_B, convex in the two components.

    PSD because a convex combination of PSD kernels is PSD, and the feature map
    is the corresponding weighted concatenation -- so the mixture is still a
    single dot-product index, just a wider one. Combines exact primitive overlap
    with learned primitive relationships.

    ``lam`` is stored as a raw parameter through a sigmoid so it stays in [0, 1]
    under unconstrained optimisation; set ``learn_lam=False`` to pin it.
    """

    def __init__(self, n_primitives: int, lam: float = 0.5,
                 rank: int | None = None, eps: float = 1e-4,
                 learn_lam: bool = True):
        super().__init__()
        self.hellinger = HellingerKernel()
        self.learned = LearnedKernel(n_primitives, rank=rank, eps=eps,
                                     normalise=True)
        lam = min(max(float(lam), 1e-6), 1 - 1e-6)
        raw = torch.logit(torch.tensor(lam))
        self.raw_lam = nn.Parameter(raw, requires_grad=learn_lam)

    @property
    def lam(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_lam)

    def feature_map(self, a: torch.Tensor) -> torch.Tensor:
        lam = self.lam
        return torch.cat([
            torch.sqrt(1 - lam) * self.hellinger.feature_map(a),
            torch.sqrt(lam) * self.learned.feature_map(a),
        ], dim=-1)


KERNELS = {
    "linear": LinearKernel,
    "hellinger": HellingerKernel,
    "learned": LearnedKernel,
    "mixture": MixtureKernel,
}


def build_kernel(name: str, n_primitives: int, **kwargs) -> Kernel:
    """Construct a kernel by name, passing only the arguments it accepts.

    Kept so a config file can name a kernel without the caller having to know
    which of them are parametric -- ``linear`` and ``hellinger`` take no
    ``n_primitives``, the other two require it.
    """
    if name not in KERNELS:
        raise ValueError(f"unknown kernel {name!r}; expected one of "
                         f"{', '.join(sorted(KERNELS))}")
    cls = KERNELS[name]
    if cls in (LinearKernel, HellingerKernel):
        return cls(**{k: v for k, v in kwargs.items() if k == "eps"})
    return cls(n_primitives=n_primitives, **kwargs)
