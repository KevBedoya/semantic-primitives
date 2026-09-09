"""Milestone 4a: does a learned kernel B beat the Hellinger affinity?

    python scripts/kernel_compare.py --n 256

Trains the same head at the same N under each kernel and compares held-out
retrieval. Orthogonal primitive coordinates have zero ordinary dot product even
when their meanings are related, and k_B = a^T B a with B = R^T R + eps I is the
design's answer to that: B_ij is literally the kernel between primitive vertices
i and j, so the model can learn that two primitives are related while the
coordinates stay orthogonal in the original inner product.

Whether that freedom is worth having is an empirical question, and the honest
answer may be no -- the Hellinger affinity is parameter-free, bounded, and
exactly normalised, which are real advantages to give up. This measures it
rather than assuming.

Also reports how far B departs from the identity. A learned kernel that stays
near diagonal has found no cross-primitive structure worth encoding, which is
informative regardless of what it does to the retrieval number.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from eval import retrieval                              # noqa: E402
from semprim.kernels import LearnedKernel, MixtureKernel  # noqa: E402
from semprim.train import load_config, train             # noqa: E402
from sweep_n import heldout_pairs_by_lang, reload_model   # noqa: E402


@torch.no_grad()
def b_matrix_stats(kernel) -> dict | None:
    """How far the learned B departs from a scaled identity.

    ``offdiag_ratio`` is the mean absolute off-diagonal entry over the mean
    diagonal entry, after normalising B to unit diagonal. Near 0 means B found
    no cross-primitive structure and the learned kernel is doing the linear
    kernel's job with extra parameters.
    """
    learned = kernel if isinstance(kernel, LearnedKernel) else \
        (kernel.learned if isinstance(kernel, MixtureKernel) else None)
    if learned is None:
        return None
    B = learned.matrix().double()
    d = B.diagonal().clamp_min(1e-12).sqrt()
    Bn = B / d.unsqueeze(0) / d.unsqueeze(1)          # unit diagonal
    n = Bn.shape[0]
    off = Bn - torch.diag(Bn.diagonal())
    return {
        "offdiag_abs_mean": float(off.abs().sum() / (n * (n - 1))),
        "offdiag_max": float(off.abs().max()),
        "eig_min": float(torch.linalg.eigvalsh(B).min()),
        "n": n,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare kernels at fixed N.")
    p.add_argument("--config", default=os.path.join("configs", "base.yaml"))
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--kernels", nargs="+",
                   default=["hellinger", "learned", "mixture"])
    p.add_argument("--out", default=os.path.join("results", "kernels"))
    p.add_argument("--epochs", type=int, default=None)
    a = p.parse_args(argv)

    base = load_config(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = heldout_pairs_by_lang(base["cache_dir"], device)
    in_dim = next(iter(pairs.values()))[0].shape[1]

    os.makedirs(a.out, exist_ok=True)
    rows = []
    for name in a.kernels:
        cfg = copy.deepcopy(base)
        cfg["model"]["n_primitives"] = int(a.n)
        cfg["kernel"] = {"name": name}
        cfg["run_name"] = f"kernel_{name}"
        if a.epochs is not None:
            cfg["train"]["epochs"] = a.epochs
        run_dir = os.path.join(a.out, name)

        print(f"{'=' * 70}\nkernel = {name}  (N={a.n})\n{'=' * 70}", flush=True)
        summary = train(cfg, run_dir)
        model, kernel = reload_model(run_dir, cfg, in_dim, device)
        ret = retrieval.evaluate(model, kernel, pairs)
        print("\n" + retrieval.format_table(ret, f"{name} retrieval P@1"))

        stats = b_matrix_stats(kernel)
        if stats:
            print(f"  B: mean |offdiag| {stats['offdiag_abs_mean']:.4f}, "
                  f"max {stats['offdiag_max']:.4f}, "
                  f"min eig {stats['eig_min']:.2e}")

        final = summary["final"] or {}
        rows.append({"kernel": name, "n": int(a.n),
                     "p_at_1": ret["mean"]["model"],
                     "baseline": ret["mean"]["baseline"],
                     "dead": final.get("dead"),
                     "active_per_text": final.get("mean_active_per_text"),
                     "b_stats": stats})
        with open(os.path.join(a.out, "kernels.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"n": a.n, "rows": rows}, f, indent=2)

    print("\nKernel comparison at N=%d, held-out P@1" % a.n)
    print(f"{'kernel':>10} {'P@1':>8} {'LaBSE':>8} {'delta':>8} {'dead':>9} "
          f"{'|offdiag|':>10}")
    for r in rows:
        od = "-" if not r["b_stats"] else f"{r['b_stats']['offdiag_abs_mean']:.4f}"
        dead = "-" if r["dead"] is None else f"{r['dead']}/{r['n']}"
        print(f"{r['kernel']:>10} {r['p_at_1']:>8.4f} {r['baseline']:>8.4f} "
              f"{r['p_at_1'] - r['baseline']:>+8.4f} {dead:>9} {od:>10}")
    print("\n|offdiag| near 0 means B learned no cross-primitive structure, so")
    print("the learned kernel is doing the linear kernel's job with more")
    print("parameters -- informative whatever it does to P@1.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
