"""Milestone 2: the reconstruction ceiling per N, with two reference points.

    python scripts/ceiling_sweep.py --n 64 256 1024

WHAT THIS NUMBER IS FOR
-----------------------
Fitting alpha directly per example, with no encoder in the way, is strictly
easier than predicting it from h(t) with an MLP -- the optimum is taken over a
superset of what the head can express. Its error therefore lower-bounds the
trained head's at the same N, and that bound is what separates two failures with
opposite remedies:

    head error >> ceiling  ->  amortisation is the problem. The mapping exists;
                               the head cannot compute it. Fix the head.
    head error ~= ceiling  ->  the bottleneck is as good as the geometry allows
                               at this N. Fix N, not the head.

Without it, a disappointing reconstruction number is uninterpretable.

TWO REFERENCE POINTS, BECAUSE THE CEILING ALONE IS ALSO UNINTERPRETABLE
-----------------------------------------------------------------------
An MSE of 0.31 means nothing in isolation, so each N is reported against:

  * **PCA at rank N** -- the best *unconstrained* linear code of the same width.
    The simplex fit is strictly more constrained (non-negative, sums to one, and
    reconstructing from a convex hull rather than a subspace), so it must do
    worse. The gap is the price of the constraint, which is the quantity the
    whole design trades interpretability for.
  * **the corpus mean** -- what a single archetype achieves. Any N whose ceiling
    is near this is not encoding anything.

Both are cheap and turn a bare number into a position on a scale.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from semprim.archetypal import archetypal_fit            # noqa: E402
from semprim.encode import load_cache                    # noqa: E402


@torch.no_grad()
def pca_error(Y: torch.Tensor, rank: int) -> float:
    """Mean squared error of the best rank-``rank`` linear reconstruction.

    Centred, so this is PCA rather than a raw truncated SVD -- the affine
    freedom is the fair comparison, since the simplex fit also gets an implicit
    offset from being able to place archetypes anywhere.
    """
    mu = Y.mean(dim=0, keepdim=True)
    Yc = (Y - mu).double()
    # Eigendecomposition of the covariance: cheaper than a full SVD when
    # d (768) is far smaller than n, which it always is here.
    cov = (Yc.t() @ Yc) / Yc.shape[0]
    evals = torch.linalg.eigvalsh(cov).flip(0)          # descending
    tail = evals[rank:].clamp_min(0).sum() if rank < evals.numel() else \
        torch.zeros((), dtype=evals.dtype)
    return float(tail)


@torch.no_grad()
def mean_error(Y: torch.Tensor) -> float:
    """MSE of reconstructing everything by the corpus mean -- the N=1 floor."""
    mu = Y.mean(dim=0, keepdim=True)
    return float(((Y - mu) ** 2).sum(dim=-1).mean())


def main(argv=None):
    p = argparse.ArgumentParser(description="Reconstruction ceiling per N.")
    p.add_argument("--cache", default=os.path.join("results", "cache", "opus7"))
    p.add_argument("--n", type=int, nargs="+", default=[64, 256, 1024])
    p.add_argument("--sample", type=int, default=50000)
    p.add_argument("--outer", type=int, default=30)
    p.add_argument("--inner", type=int, default=100)
    p.add_argument("--chunk", type=int, default=12500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join("results", "ceiling"))
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)

    device = torch.device("cuda" if (a.device == "auto" and
                                     torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))

    cache = load_cache(a.cache)
    print(f"cache: {cache.shape} {cache.dtype}  ({a.cache})")

    g = torch.Generator().manual_seed(a.seed)
    idx = torch.randperm(cache.shape[0], generator=g)[:min(a.sample,
                                                           cache.shape[0])]
    Y = torch.from_numpy(np.asarray(cache[np.sort(idx.numpy())],
                                    dtype=np.float32)).to(device)
    # Normalised, matching losses.reconstruction_loss. If the ceiling were
    # computed against unnormalised targets it would not bound the trained head
    # at all -- the two would be minimising different objectives.
    Y = F.normalize(Y, dim=-1)
    print(f"sample: {tuple(Y.shape)} on {device}, L2-normalised\n")

    floor = mean_error(Y)
    rows = []
    for n in a.n:
        t0 = time.perf_counter()
        _, _, mse = archetypal_fit(Y, n, outer=a.outer, inner=a.inner,
                                   seed=a.seed, normalize=False,
                                   chunk=a.chunk)
        pca = pca_error(Y, n)
        rows.append({"n": n, "ceiling": mse, "pca_rank_n": pca,
                     "mean_floor": floor,
                     "constraint_cost": mse - pca,
                     "seconds": time.perf_counter() - t0})
        print(f"N={n:>5}  ceiling {mse:.5f}   PCA@{n} {pca:.5f}   "
              f"cost of simplex {mse - pca:+.5f}   "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)

    print(f"\ncorpus-mean floor (N=1 equivalent): {floor:.5f}")
    print("\n" + format_table(rows, floor))

    os.makedirs(a.out, exist_ok=True)
    payload = {"cache": a.cache, "sample": int(Y.shape[0]),
               "mean_floor": floor, "rows": rows,
               "outer": a.outer, "inner": a.inner, "seed": a.seed}
    with open(os.path.join(a.out, "ceiling.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {os.path.join(a.out, 'ceiling.json')}")


def format_table(rows, floor: float) -> str:
    out = ["Reconstruction ceiling (MSE against L2-normalised LaBSE)",
           f"{'N':>6} {'ceiling':>10} {'PCA@N':>10} {'simplex cost':>13} "
           f"{'% of floor':>11}"]
    for r in rows:
        out.append(f"{r['n']:>6} {r['ceiling']:>10.5f} {r['pca_rank_n']:>10.5f} "
                   f"{r['constraint_cost']:>+13.5f} "
                   f"{100 * r['ceiling'] / max(floor, 1e-12):>10.1f}%")
    out.append("")
    out.append("ceiling      = best achievable by ANY head at this N; the trained")
    out.append("               head cannot beat it, and matching it means N is the")
    out.append("               thing to change, not the head.")
    out.append("simplex cost = ceiling - PCA@N, the price of requiring non-negative")
    out.append("               coefficients summing to one instead of a free")
    out.append("               linear code of the same width.")
    out.append(f"% of floor   = against reconstructing everything by the corpus")
    out.append(f"               mean ({floor:.5f}); near 100% means N encodes nothing.")
    return "\n".join(out)


if __name__ == "__main__":
    main()
