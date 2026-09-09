"""Is the archetype blow-up a fitting artefact, or is the objective asking for it?

    python scripts/decoder_geometry.py --n 256

THE OBSERVATION
---------------
The Milestone-2 sweep fits archetypes 9-17x further from the origin than any
real text, whose nearest actual neighbour has cosine ~0.45. Column d_i is
supposed to be "the teacher-space meaning of primitive i"; a point no text
resembles cannot serve that role, so if this is inherent the interpretability
half of the design is in trouble.

THE HYPOTHESIS
--------------
It is inherent, and it is arithmetic rather than optimisation. A convex
combination of unit vectors has norm <= 1, with equality only when the mass sits
on coincident vertices. The targets are L2-normalised, so they have norm exactly
1. A simplex code over *interpretable* (data-adjacent, unit-norm) archetypes
therefore cannot reach the target norm unless it is effectively one-hot, and
squared error charges it for the shortfall. Pushing the archetypes outward is
the only way to recover the norm while still mixing -- so the objective is
buying norm at the cost of interpretability, and getting exactly what it asked
for.

WHAT THIS SCRIPT MEASURES
-------------------------
Three fits at the same N, reported on both squared error and cosine:

  free      archetypes unconstrained (the Milestone-2 default)
  sphere    archetypes pinned to the unit sphere (the interpretable regime)

If the hypothesis holds, the sphere fit should lose badly on MSE while staying
close on *cosine* -- i.e. it recovers the direction and only misses the norm.
That would establish that the fix is to stop scoring norm at all (cosine
reconstruction, or a free per-example scale) rather than to accept
uninterpretable archetypes.

If instead the sphere fit loses on cosine too, the hypothesis is wrong: the
constraint is genuinely costing semantic content, and the tension is real rather
than an artefact of the loss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from semprim.archetypal import archetypal_fit            # noqa: E402
from semprim.encode import load_cache                    # noqa: E402


@torch.no_grad()
def report(D: torch.Tensor, A: torch.Tensor, Y: torch.Tensor) -> dict:
    R = A @ D
    return {
        "mse": float(((R - Y) ** 2).sum(dim=-1).mean()),
        "cosine": float(F.cosine_similarity(R, Y, dim=-1).mean()),
        "recon_norm_mean": float(R.norm(dim=-1).mean()),
        "archetype_radius_median": float(D.norm(dim=-1).median()),
        "archetype_radius_max": float(D.norm(dim=-1).max()),
        "nearest_text_cos_mean": float(
            torch.cat([(F.normalize(D, dim=-1)[i:i + 256]
                        @ F.normalize(Y, dim=-1).T).max(dim=1).values
                       for i in range(0, D.shape[0], 256)]).mean()),
        "active_per_text": float((A > 1e-3).float().sum(dim=-1).mean()),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Free vs sphere-constrained archetypes.")
    p.add_argument("--cache", default=os.path.join("results", "cache", "opus7"))
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--sample", type=int, default=20000)
    p.add_argument("--outer", type=int, default=30)
    p.add_argument("--inner", type=int, default=100)
    p.add_argument("--chunk", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join("results", "ceiling"))
    a = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = load_cache(a.cache)
    g = torch.Generator().manual_seed(a.seed)
    idx = torch.randperm(cache.shape[0], generator=g)[:a.sample].numpy()
    Y = torch.from_numpy(np.asarray(cache[np.sort(idx)], dtype=np.float32)).to(device)
    Y = F.normalize(Y, dim=-1)
    print(f"sample {tuple(Y.shape)} on {device}, N={a.n}\n")

    # (label, archetype constraint, free per-example scale)
    #
    # The third row is the decisive one. "sphere" pins the archetypes to the
    # data's own sphere but still scores squared error, so it is charged for a
    # norm it structurally cannot reach. "sphere+scale" pins them the same way
    # but lets a free per-example scale absorb the norm, which is identical to
    # scoring cosine instead of distance. Comparing rows 2 and 3 separates "the
    # constraint costs semantic content" from "the loss was charging for norm".
    modes = [("free", "none", False),
             ("sphere", "sphere", False),
             ("sphere+scale", "sphere", True)]

    results = {}
    for label, constrain, free_scale in modes:
        D, A, _ = archetypal_fit(Y, a.n, outer=a.outer, inner=a.inner,
                                 seed=a.seed, normalize=False, chunk=a.chunk,
                                 constrain=constrain, free_scale=free_scale)
        results[label] = report(D, A, Y)
        r = results[label]
        print(f"{label:>13}  mse {r['mse']:.5f}  cosine {r['cosine']:.4f}  "
              f"||Da|| {r['recon_norm_mean']:.3f}  "
              f"radius {r['archetype_radius_median']:.2f}  "
              f"nearest-text cos {r['nearest_text_cos_mean']:.3f}", flush=True)

    free, sph, sc = results["free"], results["sphere"], results["sphere+scale"]
    print("\n--- verdict ---")
    print(f"cosine : free {free['cosine']:.4f} | sphere {sph['cosine']:.4f} "
          f"| sphere+scale {sc['cosine']:.4f}")
    print(f"radius : free {free['archetype_radius_median']:.2f} | "
          f"sphere {sph['archetype_radius_median']:.2f} | "
          f"sphere+scale {sc['archetype_radius_median']:.2f}   "
          f"(data radius 1.00)")
    print(f"nearest-text cos : free {free['nearest_text_cos_mean']:.3f} | "
          f"sphere+scale {sc['nearest_text_cos_mean']:.3f}")

    recovered = sc["cosine"] - sph["cosine"]
    print()
    if sc["cosine"] >= free["cosine"] - 0.05:
        print(f"Interpretable archetypes match the free fit on DIRECTION once the")
        print(f"loss stops charging for norm (cosine {sc['cosine']:.4f} vs "
              f"{free['cosine']:.4f}); the sphere fit's deficit was "
              f"{recovered:+.4f} of pure norm penalty.")
        print("=> Milestone 3 should reconstruct with cosine, or give the decoder")
        print("   a free per-example scale. Keeping squared error on normalised")
        print("   targets buys norm accuracy with uninterpretable archetypes.")
    elif recovered > 0.05:
        print(f"Freeing the scale recovers {recovered:+.4f} cosine, so part of the")
        print("sphere fit's deficit was a norm penalty -- but it does not close")
        print("the gap to the free fit. Both effects are real: change the loss,")
        print("and expect interpretable archetypes still to cost accuracy.")
    else:
        print("Freeing the scale changes little. The sphere constraint costs real")
        print("semantic content, not norm bookkeeping, so the tension between")
        print("interpretable archetypes and reconstruction is genuine.")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"decoder_geometry_n{a.n}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n": a.n, "sample": int(Y.shape[0]), "results": results},
                  f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
