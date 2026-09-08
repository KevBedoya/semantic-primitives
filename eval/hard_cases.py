"""PAWS-X: the test the architecture is expected to fail.

WHY THIS IS THE LOAD-BEARING EVALUATION
---------------------------------------
A mixture over primitives has no mechanism for role binding. "The dog chased the
person" and "the person chased the dog" contain the same entities and the same
relation; if the primitive inventory is over objects and topics, both texts put
mass on the same coordinates in nearly the same proportions, and no kernel over
those coordinates can separate them. The contextual encoder upstream *can* tell
them apart -- the information is present in h(t) -- but the bottleneck discards
whatever is not expressible as a share of semantic mass.

PAWS-X is built from exactly this: high-lexical-overlap pairs where the negatives
are word-scrambled rewrites differing in argument structure. So it separates the
two hypotheses that retrieval alone confounds:

  * good retrieval + good PAWS-X -> the mixture is carrying structure;
  * good retrieval + chance PAWS-X -> it is a topic model with a kernel, which
    is a useful thing but not the thing claimed.

Run it early. If the second holds, the planned fork is several simplexes
(entities, relations, roles, polarity) with a kernel over the combination, and
finding that out at Milestone 4 is much cheaper than at Milestone 8.

Chance is the majority-class rate, reported alongside, because PAWS-X is not
balanced and an accuracy near 55% can look like signal when it is not.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def _similarities(model, kernel, h_a, h_b):
    phi_a = kernel.feature_map(model(h_a))
    phi_b = kernel.feature_map(model(h_b))
    return (phi_a * phi_b).sum(dim=-1).cpu().numpy()


@torch.no_grad()
def _baseline(h_a, h_b):
    return (F.normalize(h_a, dim=-1) * F.normalize(h_b, dim=-1)
            ).sum(dim=-1).cpu().numpy()


def _best_threshold_accuracy(sim: np.ndarray, labels: np.ndarray) -> tuple:
    """Accuracy at the best possible threshold, and that threshold.

    Deliberately optimistic: it tunes the decision boundary on the test set
    itself. That is the right choice here because the question is whether the
    *representation* separates these pairs at all, not whether a calibrated
    classifier can be fitted. If accuracy is at chance even with an oracle
    threshold, no downstream calibration will rescue it -- which is exactly the
    negative result this evaluation exists to establish cleanly.
    """
    order = np.argsort(sim)
    s, y = sim[order], labels[order]
    n_pos = y.sum()
    # Predicting "paraphrase" above a cut: accuracy as the cut sweeps upward.
    tp = n_pos - np.cumsum(y)                  # positives above each cut
    tn = np.cumsum(1 - y)                      # negatives at or below
    acc = (tp + tn) / len(y)
    best = int(np.argmax(acc))
    return float(acc[best]), float(s[best])


@torch.no_grad()
def evaluate(model, kernel, h_a: torch.Tensor, h_b: torch.Tensor,
             labels: torch.Tensor, langs=None) -> dict:
    """Accuracy at the oracle threshold, model vs frozen encoder vs chance."""
    y = labels.cpu().numpy().astype(np.int64)
    model_sim = _similarities(model, kernel, h_a, h_b)
    base_sim = _baseline(h_a, h_b)

    majority = float(max(y.mean(), 1 - y.mean()))
    m_acc, m_thr = _best_threshold_accuracy(model_sim, y)
    b_acc, _ = _best_threshold_accuracy(base_sim, y)

    out = {"overall": {"model": m_acc, "baseline": b_acc, "chance": majority,
                       "threshold": m_thr, "n": int(len(y))}}
    if langs is not None:
        langs = np.asarray(langs)
        for lang in sorted(set(langs.tolist())):
            m = langs == lang
            if m.sum() < 20:
                continue
            a_m, _ = _best_threshold_accuracy(model_sim[m], y[m])
            a_b, _ = _best_threshold_accuracy(base_sim[m], y[m])
            out[lang] = {"model": a_m, "baseline": a_b,
                         "chance": float(max(y[m].mean(), 1 - y[m].mean())),
                         "n": int(m.sum())}
    return out


def format_table(rows: dict, title: str = "PAWS-X accuracy (oracle threshold)") -> str:
    out = [title,
           f"{'lang':>8} {'n':>7} {'model':>8} {'LaBSE':>8} {'chance':>8} {'vs chance':>10}"]
    for lang, r in rows.items():
        out.append(f"{lang:>8} {r['n']:>7,} {r['model']:>8.3f} "
                   f"{r['baseline']:>8.3f} {r['chance']:>8.3f} "
                   f"{r['model'] - r['chance']:>+10.3f}")
    out.append("")
    out.append("A model column at or near the chance column means the mixture is "
               "not carrying argument structure -- see the module docstring for "
               "what follows from that.")
    return "\n".join(out)
