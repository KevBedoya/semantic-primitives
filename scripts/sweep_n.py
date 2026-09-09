"""Milestone 3: train the head at each N and plot the accuracy/interpretability trade.

    python scripts/sweep_n.py --n 64 256 1024

For each N: train from configs/base.yaml, reload the checkpoint, and evaluate
cross-lingual retrieval on the held-out split against the frozen-LaBSE baseline.
Writes a table, a trade curve, and one results directory per point.

The reload is deliberate rather than reusing the in-memory model: it exercises
the checkpoint round-trip, so a run whose weights do not survive being written
and read back fails here instead of at Milestone 4.

WHAT THE CURVE IS FOR
---------------------
The bottleneck is expected to lose retrieval accuracy against the 768-dimensional
encoder it compresses. This plots the size of that loss against N, next to the
dead-coordinate count -- because an N that scores well while leaving half its
coordinates unused is not really that N. Those two panels together are the
Milestone 3 deliverable; neither means much alone.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)                      # for the eval/ package

from eval import retrieval                              # noqa: E402
from semprim.encode import load_cache                    # noqa: E402
from semprim.kernels import build_kernel                 # noqa: E402
from semprim.model import PrimitiveModel                 # noqa: E402
from semprim.train import load_config, train             # noqa: E402


def heldout_pairs_by_lang(cache_dir: str, device) -> dict:
    """{lang: (h_en, h_xx)} over held-out pairs only, aligned row for row."""
    with open(os.path.join(cache_dir, "corpus.json"), encoding="utf-8") as f:
        corpus = json.load(f)
    cache = load_cache(cache_dir)
    H = torch.from_numpy(np.asarray(cache, dtype=np.float32)).to(device)

    pairs = np.asarray(corpus["pairs"])
    held = np.asarray(corpus["heldout"], dtype=bool)
    langs = np.asarray(corpus["pair_langs"])

    out = {}
    for lang in sorted(set(langs.tolist())):
        sel = held & (langs == lang)
        if sel.sum() < 2:
            continue
        idx = pairs[sel]
        out[lang] = (H[torch.from_numpy(idx[:, 0]).to(device)],
                     H[torch.from_numpy(idx[:, 1]).to(device)])
    return out


def reload_model(out_dir: str, cfg: dict, in_dim: int, device):
    n = int(cfg["model"]["n_primitives"])
    model = PrimitiveModel(
        in_dim=in_dim, teacher_dim=in_dim,
        hidden=int(cfg["model"].get("hidden", 1024)), n_primitives=n,
        alpha=1.0, tau_init=float(cfg["model"].get("tau_init", 1.0)),
        learn_tau=bool(cfg["model"].get("learn_tau", True))).to(device)
    kernel = build_kernel(cfg["kernel"]["name"], n,
                          **{k: v for k, v in cfg["kernel"].items()
                             if k != "name"}).to(device)
    ckpt = torch.load(os.path.join(out_dir, "checkpoint.pt"),
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    kernel.load_state_dict(ckpt["kernel"])
    # The trained alpha, not the constructor default: entmax alpha is annealed
    # during training and is not a learned parameter, so it is not in the state
    # dict. Evaluating a run trained to alpha=2.0 at alpha=1.0 would score a
    # dense model that was never trained.
    final_alpha = cfg["train"].get("alpha_schedule", [[0, 1.0]])
    model.bottleneck.set_alpha(float(sorted(final_alpha,
                                            key=lambda p: p[0])[-1][1]))
    model.eval()
    return model, kernel


def main(argv=None):
    p = argparse.ArgumentParser(description="Milestone 3 N sweep.")
    p.add_argument("--config", default=os.path.join("configs", "base.yaml"))
    p.add_argument("--n", type=int, nargs="+", default=[64, 256, 1024])
    p.add_argument("--out", default=os.path.join("results", "sweep"))
    p.add_argument("--epochs", type=int, default=None,
                   help="override train.epochs; for validating the wiring "
                        "cheaply before committing the full run")
    p.add_argument("--archetypal-outer", type=int, default=None,
                   help="override archetypal.outer, likewise")
    a = p.parse_args(argv)

    base = load_config(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = heldout_pairs_by_lang(base["cache_dir"], device)
    in_dim = next(iter(pairs.values()))[0].shape[1]
    print(f"held-out eval: {len(pairs)} languages, "
          f"{sum(v[0].shape[0] for v in pairs.values()):,} pairs, dim {in_dim}\n")

    os.makedirs(a.out, exist_ok=True)
    rows = []
    for n in a.n:
        cfg = copy.deepcopy(base)
        cfg["model"]["n_primitives"] = int(n)
        cfg["run_name"] = f"n{n}"
        if a.epochs is not None:
            cfg["train"]["epochs"] = a.epochs
        if a.archetypal_outer is not None:
            cfg.setdefault("archetypal", {})["outer"] = a.archetypal_outer
        run_dir = os.path.join(a.out, f"n{n}")

        print(f"{'=' * 70}\nN = {n}\n{'=' * 70}", flush=True)
        summary = train(cfg, run_dir)

        model, kernel = reload_model(run_dir, cfg, in_dim, device)
        ret = retrieval.evaluate(model, kernel, pairs)
        print("\n" + retrieval.format_table(ret, f"N={n} retrieval P@1"))

        final = summary["final"] or {}
        rows.append({
            "n": int(n),
            "p_at_1": ret["mean"]["model"],
            "baseline": ret["mean"]["baseline"],
            "per_lang": {k: v for k, v in ret.items() if k != "mean"},
            "dead": final.get("dead"),
            "active_per_text": final.get("mean_active_per_text"),
            "reconstruct": final.get("reconstruct"),
            "ceiling": summary.get("archetypal_ceiling"),
            "matched_ceiling": (summary.get("sparsity_matched_ceiling") or {}
                                ).get("ceiling"),
            "matched_k": (summary.get("sparsity_matched_ceiling") or {}
                          ).get("k"),
            "tau": final.get("tau"),
            "alpha": final.get("alpha"),
        })
        with open(os.path.join(a.out, "sweep.json"), "w", encoding="utf-8") as f:
            json.dump({"config": a.config, "rows": rows}, f, indent=2)

    print("\n" + format_table(rows))
    plot(rows, os.path.join(a.out, "trade_curve.png"))
    with open(os.path.join(a.out, "sweep.json"), "w", encoding="utf-8") as f:
        json.dump({"config": a.config, "rows": rows}, f, indent=2)
    print(f"\nwrote {a.out}")


def format_table(rows) -> str:
    base = rows[0]["baseline"] if rows else float("nan")
    out = ["Milestone 3: bottleneck vs frozen LaBSE, held-out P@1",
           f"{'N':>6} {'P@1':>8} {'LaBSE':>8} {'delta':>8} {'dead':>9} "
           f"{'act/text':>9} {'recon':>9} {'k-ceil':>9} {'ratio':>7}"]
    for r in rows:
        dead = "-" if r["dead"] is None else f"{r['dead']}/{r['n']}"
        act = "-" if r["active_per_text"] is None else f"{r['active_per_text']:.1f}"
        rec = "-" if r["reconstruct"] is None else f"{r['reconstruct']:.5f}"
        mc = r.get("matched_ceiling")
        ceil = "-" if mc is None else f"{mc:.5f}"
        ratio = ("-" if (mc is None or r["reconstruct"] is None)
                 else f"{r['reconstruct'] / max(mc, 1e-12):.2f}x")
        out.append(f"{r['n']:>6} {r['p_at_1']:>8.4f} {r['baseline']:>8.4f} "
                   f"{r['p_at_1'] - r['baseline']:>+8.4f} {dead:>9} "
                   f"{act:>9} {rec:>9} {ceil:>9} {ratio:>7}")
    out.append("")
    out.append(f"LaBSE cosine on the same pairs: {base:.4f}. The bottleneck is")
    out.append("expected to lose; the deliverable is the size of the loss.")
    out.append("")
    out.append("k-ceil is the ceiling fitted at the head's OWN support size, not")
    out.append("the dense one. A sparse head compared against a dense fit is not")
    out.append("compared against a bound at all. Both are computed, so the size")
    out.append("of that mistake is shown rather than asserted:")
    for r in rows:
        mc, dc, rec = r.get("matched_ceiling"), r.get("ceiling"), r["reconstruct"]
        if None in (mc, dc, rec):
            continue
        out.append(f"    N={r['n']:<5} matched(k={r.get('matched_k')}) "
                   f"{rec / max(mc, 1e-12):>7.2f}x   "
                   f"dense {rec / max(dc, 1e-12):>10.2f}x")
    out.append("Ratio near 1 means N is the thing to change; a ratio that GROWS")
    out.append("with N means the head is failing to exploit the larger inventory,")
    out.append("which the dead count corroborates.")
    return "\n".join(out)


def plot(rows, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["n"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(ns, [r["p_at_1"] for r in rows], marker="o", label="bottleneck")
    ax1.axhline(rows[0]["baseline"], ls="--", color="k",
                label=f"frozen LaBSE ({rows[0]['baseline']:.3f})")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ns)
    ax1.set_xticklabels([str(n) for n in ns])
    ax1.set_xlabel("N (primitives)")
    ax1.set_ylabel("held-out P@1")
    ax1.set_title("Retrieval accuracy vs N")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Second panel: an N that scores well while leaving coordinates unused is
    # not really that N, so usage is plotted next to accuracy rather than
    # buried in a log.
    used = [r["n"] - (r["dead"] or 0) for r in rows]
    ax2.plot(ns, used, marker="s", color="tab:orange", label="coordinates used")
    ax2.plot(ns, ns, ls=":", color="gray", label="N (all used)")
    act = [r["active_per_text"] for r in rows]
    if all(v is not None for v in act):
        ax2.plot(ns, act, marker="^", color="tab:green",
                 label="active per text")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=2)
    ax2.set_xticks(ns)
    ax2.set_xticklabels([str(n) for n in ns])
    ax2.set_xlabel("N (primitives)")
    ax2.set_ylabel("coordinates")
    ax2.set_title("Inventory usage vs N")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
