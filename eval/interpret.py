"""Interpretability diagnostics -- what turns "compressed embedding" into a claim.

Retrieval and STS say the bottleneck kept useful information. They say nothing
about whether the coordinates *mean* anything, and a topic model with a good
kernel would score identically. These four diagnostics are what separates the
two, and they are the actual Milestone-4 deliverable:

  1. top_activating   -- what does coordinate i fire on?
  2. coherence        -- is what it fires on tight, or a grab bag?
  3. cross_language   -- is coordinate i the same concept in all 7 languages,
                         or is it detecting orthography?
  4. ablation         -- does removing coordinate i cost anything, and what?

Diagnostic 3 is the one that matters most for the central claim, and diagnostic 2
is why it needs a control: any set of top-k items looks somewhat coherent, so
coherence is only meaningful against random sets from the same corpus.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def all_alphas(model, H: torch.Tensor, batch: int = 4096) -> torch.Tensor:
    return torch.cat([model(H[i:i + batch]) for i in range(0, H.shape[0], batch)])


@torch.no_grad()
def top_activating(alpha: torch.Tensor, texts, k: int = 50) -> dict:
    """The k texts that put the most mass on each coordinate.

    The first thing to read for any coordinate. If the top-k for a coordinate has
    no describable common thread, that coordinate is not a primitive whatever the
    retrieval numbers say.
    """
    out = {}
    top = alpha.topk(k=min(k, alpha.shape[0]), dim=0)
    for i in range(alpha.shape[1]):
        idx = top.indices[:, i].tolist()
        out[i] = [(texts[j], float(top.values[r, i])) for r, j in enumerate(idx)]
    return out


@torch.no_grad()
def coherence(alpha: torch.Tensor, H: torch.Tensor, k: int = 50,
              n_control: int = 20, seed: int = 0) -> dict:
    """Mean pairwise teacher-cosine within each coordinate's top-k, vs a control.

    The control is essential and easy to omit. Any k embeddings drawn from a
    corpus have some positive mean cosine, so a raw coherence of 0.4 is
    uninterpretable on its own. What matters is the lift over random sets of the
    same size from the same corpus.
    """
    Hn = F.normalize(H, dim=-1)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def mean_pairwise(idx):
        V = Hn[idx]
        S = V @ V.T
        n = S.shape[0]
        if n < 2:
            return float("nan")
        return float((S.sum() - S.diagonal().sum()) / (n * (n - 1)))

    controls = [mean_pairwise(torch.randperm(H.shape[0], generator=g)[:k].to(H.device))
                for _ in range(n_control)]
    baseline = float(np.mean(controls))

    top = alpha.topk(k=min(k, alpha.shape[0]), dim=0).indices
    out = {}
    for i in range(alpha.shape[1]):
        c = mean_pairwise(top[:, i])
        out[i] = {"coherence": c, "lift": c - baseline}
    out["_control"] = {"coherence": baseline, "lift": 0.0,
                       "std": float(np.std(controls))}
    return out


@torch.no_grad()
def cross_language(alpha: torch.Tensor, langs, k: int = 50) -> dict:
    """Language composition of each coordinate's top-k.

    THE test for the central claim. A coordinate whose top-k is 90% one language
    is detecting that language -- script, tokenisation, corpus artefacts -- not a
    language-independent concept, and it will still contribute to retrieval,
    which is why retrieval cannot catch this.

    ``entropy_ratio`` is the entropy of the language distribution over the
    coordinate's top-k, divided by the entropy of the corpus language
    distribution. 1.0 means the coordinate is used exactly as evenly across
    languages as the corpus allows; near 0 means it is a language detector.
    """
    langs = np.asarray(langs)
    uniq = sorted(set(langs.tolist()))
    lang_id = {l: i for i, l in enumerate(uniq)}
    ids = np.array([lang_id[l] for l in langs])

    corpus_counts = np.bincount(ids, minlength=len(uniq)).astype(np.float64)
    corpus_p = corpus_counts / corpus_counts.sum()
    corpus_H = -(corpus_p * np.log(corpus_p + 1e-12)).sum()

    top = alpha.topk(k=min(k, alpha.shape[0]), dim=0).indices.cpu().numpy()
    out = {}
    for i in range(alpha.shape[1]):
        counts = np.bincount(ids[top[:, i]], minlength=len(uniq)).astype(np.float64)
        p = counts / counts.sum()
        H = -(p * np.log(p + 1e-12)).sum()
        out[i] = {
            "entropy_ratio": float(H / corpus_H) if corpus_H > 0 else float("nan"),
            "dominant_lang": uniq[int(counts.argmax())],
            "dominant_share": float(counts.max() / counts.sum()),
        }
    return out


@torch.no_grad()
def paired_consistency(model, h_a: torch.Tensor, h_b: torch.Tensor,
                       k: int = 5, seed: int = 0) -> dict:
    """Do a text and its translation activate the SAME coordinates?

    The most direct test of the central claim. Language-independent primitives
    means coordinate i denotes the same thing whichever language expresses it,
    so for a held-out translation pair the two sides should land on the same
    coordinates -- not merely end up nearby under the kernel, which the
    contrastive loss could achieve by any means.

    Reported three ways, because each alone is misleading:

      * ``top1``     -- fraction of pairs whose argmax coordinate agrees. Strict,
                        and the number to quote.
      * ``overlap``  -- mean |top-k(a) ∩ top-k(b)| / k. Forgiving of ties among
                        near-equal coordinates, which are common on a simplex.
      * ``*_control`` -- the same statistics over *mismatched* pairs. Essential:
                        with 21 of 256 coordinates active and a skewed usage
                        distribution, unrelated texts already share coordinates
                        at some rate. Only the lift over that rate is evidence.

    Note this is what L_align optimises, so a good score is not a surprise --
    it is a check that the objective transferred to held-out data rather than a
    discovery. A *bad* score here would be damning.
    """
    a = model(h_a)
    b = model(h_b)
    n = a.shape[0]

    def stats(x, y):
        top1 = (x.argmax(dim=-1) == y.argmax(dim=-1)).float().mean().item()
        kk = min(k, x.shape[-1])
        tx = x.topk(kk, dim=-1).indices
        ty = y.topk(kk, dim=-1).indices
        # Set overlap per row, without materialising an (n, N) one-hot.
        hit = (tx.unsqueeze(2) == ty.unsqueeze(1)).any(dim=2).float().sum(dim=1)
        return top1, (hit / kk).mean().item()

    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=g).to(b.device)
    t1, ov = stats(a, b)
    t1c, ovc = stats(a, b[perm])
    return {"top1": t1, "top1_control": t1c, "top1_lift": t1 - t1c,
            "overlap": ov, "overlap_control": ovc, "overlap_lift": ov - ovc,
            "k": k, "n": n}


@torch.no_grad()
def ablate(model, kernel, h_query: torch.Tensor, h_cand: torch.Tensor,
           coordinate: int) -> float:
    """P@1 with one coordinate zeroed and the rest renormalised.

    Renormalisation is required, not cosmetic: dropping a coordinate without it
    leaves a sub-probability vector, and every kernel here assumes the simplex.
    The delta against the unablated score is what a coordinate is *worth*.
    """
    from .retrieval import precision_at_1

    def masked(h):
        a = model(h).clone()
        a[:, coordinate] = 0.0
        return a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    return precision_at_1(kernel(masked(h_query), masked(h_cand)))


@torch.no_grad()
def ablation_sweep(model, kernel, h_query, h_cand, coordinates=None) -> dict:
    """Retrieval cost of removing each coordinate. Returns {i: delta_P@1}.

    A genuine inventory should degrade gracefully and unevenly -- some primitives
    carry more than others. A flat profile near zero means the coordinates are
    redundant, which is itself a finding about N being too large.
    """
    from .retrieval import precision_at_1

    full = precision_at_1(kernel(model(h_query), model(h_cand)))
    n = model.n_primitives if coordinates is None else len(coordinates)
    coords = range(model.n_primitives) if coordinates is None else coordinates
    out = {"_full": full}
    for i in coords:
        out[int(i)] = ablate(model, kernel, h_query, h_cand, int(i)) - full
    return out
