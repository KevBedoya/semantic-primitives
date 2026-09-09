"""Milestone 4b: the go/no-go on "reusable semantic primitives".

    python scripts/interpret_report.py --run results/sweep/n256

Retrieval says the bottleneck kept useful information. It says nothing about
whether the coordinates *mean* anything, and a topic model with a good kernel
would score identically. These four diagnostics are what separate the two, and
they are the actual deliverable of this project -- the retrieval table is
supporting evidence.

THRESHOLDS ARE FIXED BELOW, BEFORE ANY RESULTS WERE SEEN
--------------------------------------------------------
Written and committed while the training runs were still in flight. That
ordering matters more than the specific values: a bar chosen after seeing the
numbers is not a bar. They are judgement calls and are argued for individually,
not derived, so disagree with them explicitly rather than by quietly moving
them.

A "no" here is a real possible outcome and not a failure of the work. Locatello
et al. (2019) is the standing reason to expect it: orthogonality and convexity
constrain geometry, not meaning, and a unique human-interpretable inventory
needs supervision or extra assumptions that this stage deliberately does not
supply.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from eval import interpret, retrieval                   # noqa: E402
from semprim.encode import load_cache                    # noqa: E402
from sweep_n import heldout_pairs_by_lang, reload_model   # noqa: E402

# --------------------------------------------------------------- thresholds
#
# LANG_ENTROPY_MIN: median language-entropy ratio over *used* coordinates.
#   A coordinate whose top-k activations come from one language is detecting
#   orthography or corpus artefacts, and it will still contribute to retrieval,
#   which is exactly why retrieval cannot catch it. 0.70 is lenient on purpose:
#   the corpus is 50% English by construction (pairs are pivoted through it), so
#   a genuinely language-neutral coordinate still looks English-heavy, and the
#   ratio is normalised against that skew rather than against uniformity.
LANG_ENTROPY_MIN = 0.70

# PAIRED_TOP1_LIFT_MIN: agreement of the argmax coordinate between a text and
#   its translation, over the mismatched-pair control. This is what L_align
#   optimises, so a good score is a check that it transferred to held-out data,
#   not a discovery -- but a bad score would be damning, and the control is
#   essential because ~21 active of 256 means unrelated texts already collide.
PAIRED_TOP1_LIFT_MIN = 0.30

# COHERENCE_LIFT_MIN: mean teacher-cosine within a coordinate's top-k, over
#   random sets of the same size. Any k texts have some positive mean cosine, so
#   only the lift is evidence. 0.10 is modest: it asks that a coordinate's
#   activations be noticeably tighter than chance, not that they be a clean
#   human concept.
COHERENCE_LIFT_MIN = 0.10

# ABLATION_UNEVENNESS_MIN: std/mean of the per-coordinate retrieval cost. A
#   genuine inventory should degrade unevenly -- some primitives carry more than
#   others. A flat profile means the coordinates are redundant, which is a
#   finding about N rather than about meaning.
ABLATION_UNEVENNESS_MIN = 0.5


def load_run(run_dir: str, device):
    with open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pairs = heldout_pairs_by_lang(cfg["cache_dir"], device)
    in_dim = next(iter(pairs.values()))[0].shape[1]
    model, kernel = reload_model(run_dir, cfg, in_dim, device)
    return cfg, model, kernel, pairs


def main(argv=None):
    p = argparse.ArgumentParser(description="Interpretability go/no-go.")
    p.add_argument("--run", default=os.path.join("results", "sweep", "n256"))
    p.add_argument("--sample", type=int, default=60000)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--ablate", type=int, default=32,
                   help="how many coordinates to ablate (most-used first)")
    p.add_argument("--out", default=os.path.join("results", "interpret"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model, kernel, pairs = load_run(a.run, device)
    n_prim = model.n_primitives
    print(f"run {a.run}: N={n_prim}, kernel={cfg['kernel']['name']}, "
          f"entmax alpha={model.bottleneck.alpha}\n")

    with open(os.path.join(cfg["cache_dir"], "corpus.json"), encoding="utf-8") as f:
        corpus = json.load(f)
    cache = load_cache(cfg["cache_dir"])

    g = torch.Generator().manual_seed(a.seed)
    idx = np.sort(torch.randperm(len(corpus["texts"]),
                                 generator=g)[:a.sample].numpy())
    texts = [corpus["texts"][i] for i in idx]
    langs = [corpus["langs"][i] for i in idx]
    H = torch.from_numpy(np.asarray(cache[idx], dtype=np.float32)).to(device)
    alpha = interpret.all_alphas(model, H)
    used = (alpha.max(dim=0).values > 1e-3)
    print(f"sample {len(texts):,} texts; {int(used.sum())}/{n_prim} "
          f"coordinates used\n")

    report = {"run": a.run, "n_primitives": n_prim,
              "kernel": cfg["kernel"]["name"],
              "coordinates_used": int(used.sum()),
              "thresholds": {
                  "lang_entropy_min": LANG_ENTROPY_MIN,
                  "paired_top1_lift_min": PAIRED_TOP1_LIFT_MIN,
                  "coherence_lift_min": COHERENCE_LIFT_MIN,
                  "ablation_unevenness_min": ABLATION_UNEVENNESS_MIN}}

    # --- 1. cross-language ------------------------------------------------
    xl = interpret.cross_language(alpha, langs, k=a.topk)
    ratios = np.array([xl[i]["entropy_ratio"] for i in range(n_prim)
                       if used[i]])
    shares = np.array([xl[i]["dominant_share"] for i in range(n_prim)
                       if used[i]])
    report["cross_language"] = {
        "median_entropy_ratio": float(np.median(ratios)),
        "frac_below_half": float((ratios < 0.5).mean()),
        "median_dominant_share": float(np.median(shares)),
        "frac_dominant_over_80pct": float((shares > 0.8).mean()),
    }
    print("1. CROSS-LANGUAGE  (are coordinates language detectors?)")
    print(f"   median entropy ratio   {np.median(ratios):.3f} "
          f"(threshold >= {LANG_ENTROPY_MIN})")
    print(f"   coords below 0.5       {(ratios < 0.5).mean():.1%}")
    print(f"   median dominant share  {np.median(shares):.3f}")
    print(f"   coords >80% one lang   {(shares > 0.8).mean():.1%}\n")

    # --- 2. paired consistency -------------------------------------------
    h_a = torch.cat([v[0] for v in pairs.values()])
    h_b = torch.cat([v[1] for v in pairs.values()])
    pc = interpret.paired_consistency(model, h_a, h_b, k=5, seed=a.seed)
    report["paired_consistency"] = pc
    print("2. PAIRED CONSISTENCY  (does a translation hit the same coordinates?)")
    print(f"   top-1 agreement        {pc['top1']:.3f}  "
          f"control {pc['top1_control']:.3f}  "
          f"lift {pc['top1_lift']:+.3f} (threshold >= {PAIRED_TOP1_LIFT_MIN})")
    print(f"   top-5 overlap          {pc['overlap']:.3f}  "
          f"control {pc['overlap_control']:.3f}  "
          f"lift {pc['overlap_lift']:+.3f}\n")

    # --- 3. coherence ------------------------------------------------------
    coh = interpret.coherence(alpha, H, k=a.topk, seed=a.seed)
    lifts = np.array([coh[i]["lift"] for i in range(n_prim) if used[i]])
    report["coherence"] = {
        "control": coh["_control"]["coherence"],
        "control_std": coh["_control"]["std"],
        "median_lift": float(np.median(lifts)),
        "frac_positive": float((lifts > 0).mean()),
    }
    print("3. COHERENCE  (is what a coordinate fires on tighter than chance?)")
    print(f"   random control cosine  {coh['_control']['coherence']:.3f} "
          f"(sd {coh['_control']['std']:.3f})")
    print(f"   median lift            {np.median(lifts):+.3f} "
          f"(threshold >= {COHERENCE_LIFT_MIN})")
    print(f"   coords with lift > 0   {(lifts > 0).mean():.1%}\n")

    # --- 4. ablation -------------------------------------------------------
    order = alpha.mean(dim=0).argsort(descending=True)
    coords = [int(c) for c in order[:a.ablate] if used[c]]
    abl = interpret.ablation_sweep(model, kernel, h_a, h_b, coordinates=coords)
    deltas = np.array([abl[c] for c in coords])
    uneven = float(deltas.std() / max(abs(deltas.mean()), 1e-9))
    report["ablation"] = {
        "full_p_at_1": abl["_full"], "n_ablated": len(coords),
        "mean_delta": float(deltas.mean()), "min_delta": float(deltas.min()),
        "unevenness": uneven,
    }
    print(f"4. ABLATION  (does removing a coordinate cost anything, unevenly?)")
    print(f"   full P@1               {abl['_full']:.4f}")
    print(f"   mean delta over {len(coords):>3} coords {deltas.mean():+.5f}")
    print(f"   worst single coord     {deltas.min():+.5f}")
    print(f"   unevenness (sd/mean)   {uneven:.2f} "
          f"(threshold >= {ABLATION_UNEVENNESS_MIN})\n")

    # --- verdict -----------------------------------------------------------
    checks = {
        "cross_language": np.median(ratios) >= LANG_ENTROPY_MIN,
        "paired_consistency": pc["top1_lift"] >= PAIRED_TOP1_LIFT_MIN,
        "coherence": np.median(lifts) >= COHERENCE_LIFT_MIN,
        "ablation_unevenness": uneven >= ABLATION_UNEVENNESS_MIN,
    }
    report["checks"] = {k: bool(v) for k, v in checks.items()}
    report["verdict"] = "go" if all(checks.values()) else "partial" \
        if any(checks.values()) else "no-go"

    print("=" * 62)
    print("VERDICT on 'reusable, language-independent primitives'")
    for name, ok in checks.items():
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"   => {report['verdict'].upper()}")
    print("=" * 62)
    print("A 'no' is a real outcome, not a failed experiment: orthogonality and")
    print("convexity constrain geometry, not meaning (Locatello et al., 2019).")

    # --- qualitative sample ------------------------------------------------
    top = interpret.top_activating(alpha, texts, k=8)
    sample_coords = coords[:6]
    lines = []
    for c in sample_coords:
        lines.append(f"\n--- coordinate {c} "
                     f"(entropy ratio {xl[c]['entropy_ratio']:.2f}, "
                     f"dominant {xl[c]['dominant_lang']} "
                     f"{xl[c]['dominant_share']:.0%}, "
                     f"coherence lift {coh[c]['lift']:+.3f}) ---")
        for text, val in top[c][:8]:
            lines.append(f"  {val:.3f}  {text[:96]}")
    print("\n".join(lines))
    report["samples"] = {int(c): [t for t, _ in top[c][:8]]
                         for c in sample_coords}

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "interpret.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {os.path.join(a.out, 'interpret.json')}")


if __name__ == "__main__":
    main()
