"""The frozen-LaBSE baseline: the row every later table has to carry.

    python scripts/baseline_table.py

Measured on the held-out pairs only, which no training run sees. Both directions
are reported (en->xx and xx->en) because they are not the same task and the gap
between them is informative: retrieval is asymmetric whenever one side's
candidate pool is more confusable than the other's.

This is the number the bottleneck will be compared against, and it is expected
to *win*. Recording it before any bottleneck exists is deliberate -- a baseline
computed after the fact, once the shape of the model's results is known, is a
baseline nobody can fully trust.
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

from semprim.encode import load_cache                    # noqa: E402


def load_corpus(cache_dir: str) -> dict:
    with open(os.path.join(cache_dir, "corpus.json"), encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def p_at_1(q: torch.Tensor, c: torch.Tensor) -> float:
    """Cosine retrieval precision@1 with the correct answer on the diagonal."""
    scores = F.normalize(q, dim=-1) @ F.normalize(c, dim=-1).T
    correct = scores.argmax(dim=1) == torch.arange(q.shape[0], device=q.device)
    return float(correct.float().mean())


def main(argv=None):
    p = argparse.ArgumentParser(description="Frozen-encoder baseline table.")
    p.add_argument("--cache", default=os.path.join("results", "cache", "opus7"))
    p.add_argument("--out", default=os.path.join("results", "baseline"))
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)

    device = torch.device("cuda" if (a.device == "auto" and
                                     torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))

    corpus = load_corpus(a.cache)
    cache = load_cache(a.cache, texts=corpus["texts"])
    H = torch.from_numpy(np.asarray(cache, dtype=np.float32)).to(device)
    print(f"cache: {tuple(H.shape)} on {device}")

    pairs = np.asarray(corpus["pairs"])
    held = np.asarray(corpus["heldout"], dtype=bool)
    pair_langs = np.asarray(corpus["pair_langs"])

    rows = {}
    for lang in sorted(set(pair_langs.tolist())):
        sel = held & (pair_langs == lang)
        if sel.sum() < 2:
            continue
        idx = pairs[sel]
        q = H[torch.from_numpy(idx[:, 0]).to(device)]      # en
        c = H[torch.from_numpy(idx[:, 1]).to(device)]      # xx
        rows[lang] = {"en_to_xx": p_at_1(q, c),
                      "xx_to_en": p_at_1(c, q),
                      "n": int(sel.sum())}
        print(f"  {lang}: n={rows[lang]['n']:,}  "
              f"en->{lang} {rows[lang]['en_to_xx']:.4f}  "
              f"{lang}->en {rows[lang]['xx_to_en']:.4f}", flush=True)

    rows["mean"] = {
        "en_to_xx": float(np.mean([r["en_to_xx"] for k, r in rows.items()
                                   if k != "mean"])),
        "xx_to_en": float(np.mean([r["xx_to_en"] for k, r in rows.items()
                                   if k != "mean"])),
        "n": int(sum(r["n"] for k, r in rows.items() if k != "mean")),
    }

    print("\nFrozen LaBSE cosine, held-out pairs, P@1")
    print(f"{'lang':>6} {'n':>7} {'en->xx':>9} {'xx->en':>9}")
    for lang, r in rows.items():
        print(f"{lang:>6} {r['n']:>7,} {r['en_to_xx']:>9.4f} "
              f"{r['xx_to_en']:>9.4f}")
    print("\nThis is the row every bottleneck result must be shown against.")
    print("The bottleneck is expected to lose against it; the deliverable is")
    print("the size of that loss as a function of N and sparsity.")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "labse_baseline.json"), "w",
              encoding="utf-8") as f:
        json.dump({"cache": a.cache, "metric": "cosine P@1",
                   "split": "heldout", "rows": rows}, f, indent=2)
    print(f"\nwrote {os.path.join(a.out, 'labse_baseline.json')}")


if __name__ == "__main__":
    main()
