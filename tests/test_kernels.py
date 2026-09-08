"""Kernel invariants.

The feature-map identity is the one that carries weight downstream: retrieval
stores feature_map(alpha) in a dot-product index, so if k and its feature map
ever disagree, every retrieval number is computed with a different kernel than
the one that was trained, and nothing anywhere would raise.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import pytest
import torch

from semprim.kernels import (HellingerKernel, LearnedKernel, LinearKernel,
                             MixtureKernel, build_kernel)

TOL = 1e-6
N = 12


def random_simplex(n, dim, seed=0, sparse=False):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, dim, generator=g)
    a = torch.softmax(z, dim=-1)
    if sparse:
        # Zero out most coordinates and renormalise: exercises the exact-zero
        # regime that entmax produces and that sqrt/normalisation must survive.
        keep = torch.zeros_like(a)
        idx = torch.topk(a, k=3, dim=-1).indices
        keep.scatter_(-1, idx, 1.0)
        a = a * keep
        a = a / a.sum(dim=-1, keepdim=True)
    return a.double()


ALL = [
    ("linear", lambda: LinearKernel()),
    ("hellinger", lambda: HellingerKernel()),
    ("learned", lambda: LearnedKernel(N, rank=5)),
    ("mixture", lambda: MixtureKernel(N, lam=0.3, rank=5)),
]


@pytest.mark.parametrize("name,make", ALL, ids=[n for n, _ in ALL])
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_feature_map_reproduces_the_kernel(name, make, sparse):
    k = make().double()
    a = random_simplex(7, N, seed=1, sparse=sparse)
    b = random_simplex(5, N, seed=2, sparse=sparse)
    direct = k(a, b)
    viamap = k.feature_map(a) @ k.feature_map(b).T
    assert torch.allclose(direct, viamap, atol=TOL), \
        f"{name}: feature map disagrees by {(direct - viamap).abs().max():.3e}"


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_hellinger_is_normalised_and_bounded(sparse):
    k = HellingerKernel().double()
    a = random_simplex(16, N, seed=3, sparse=sparse)
    diag = k(a, a).diagonal()
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5), \
        "k(t, t) must be exactly 1 because sum(alpha) = 1"
    vals = k(a, random_simplex(16, N, seed=4, sparse=sparse))
    assert vals.min() >= -TOL and vals.max() <= 1 + 1e-5


def test_learned_kernel_matrix_is_psd():
    k = LearnedKernel(N, rank=4, eps=1e-4).double()
    with torch.no_grad():
        k.R.normal_(0, 1.0)          # not the small init -- stress the claim
    eig = torch.linalg.eigvalsh(k.matrix())
    assert eig.min() >= k.eps - 1e-9, \
        f"B must be PSD with min eigenvalue >= eps; got {eig.min():.3e}"


def test_learned_kernel_matrix_matches_the_bilinear_form():
    """B_ij is the kernel between primitive vertices i and j -- the whole point.

    Without normalisation, k(e_i, e_j) must be exactly B_ij; that identity is
    what makes B readable as a table of learned primitive relationships.
    """
    k = LearnedKernel(N, rank=6, normalise=False).double()
    with torch.no_grad():
        k.R.normal_(0, 0.5)
    eye = torch.eye(N, dtype=torch.float64)
    assert torch.allclose(k(eye, eye), k.matrix(), atol=TOL)


def test_kernels_are_symmetric():
    for name, make in ALL:
        k = make().double()
        a = random_simplex(9, N, seed=5)
        m = k(a, a)
        assert torch.allclose(m, m.T, atol=TOL), f"{name} is not symmetric"


def test_build_kernel_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown kernel"):
        build_kernel("cosine", N)
