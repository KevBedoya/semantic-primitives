"""Training signals for the simplex bottleneck.

    L = lam_c L_contrast + lam_a L_align + lam_r L_reconstruct
        + lam_g L_ground + lam_s L_sparsity (+ lam_d L_diversity)

Each term is a plain function so train.py can switch any of them off with a zero
weight and the config stays the record of what actually ran.

TWO COLLAPSE MODES, NOT ONE
---------------------------
*Representational collapse* -- every text maps to the same alpha -- is what
L_align invites and what L_contrast prevents. That much is in the brief.

*Primitive collapse* is separate and is not prevented by any of the above: the
model can satisfy alignment and discrimination perfectly while using only k << N
coordinates and leaving the rest permanently at zero. Nothing in the loss
notices, the metrics look fine, and the "inventory of N primitives" is quietly an
inventory of k. ``dead_coordinates`` measures it and ``diversity_loss``
corrects it -- but only on demand. Real primitives need not occur equally often,
so forcing uniform usage is itself a distortion; this is a corrective for a
diagnosed problem, not a default.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .kernels import Kernel

_EPS = 1e-12


# --------------------------------------------------------------------- align
def align_loss(alpha_a: torch.Tensor, alpha_b: torch.Tensor,
               geometry: str = "l2") -> torch.Tensor:
    """Pull meaning-equivalent texts (translations, paraphrases) together.

    ``geometry="l2"`` is the brief's ||a - a+||^2.

    ``geometry="hellinger"`` is ||sqrt(a) - sqrt(a+)||^2, the squared Hellinger
    distance. Worth testing against l2 because it is the metric induced by the
    kernel retrieval actually scores with: aligning in one geometry and ranking
    in another means the training signal and the evaluation disagree about what
    "close" means. Costs one line to switch.
    """
    if geometry == "l2":
        return ((alpha_a - alpha_b) ** 2).sum(dim=-1).mean()
    if geometry == "hellinger":
        ra = torch.sqrt(alpha_a.clamp_min(_EPS))
        rb = torch.sqrt(alpha_b.clamp_min(_EPS))
        return ((ra - rb) ** 2).sum(dim=-1).mean()
    raise ValueError(f"unknown geometry {geometry!r}; expected 'l2' or 'hellinger'")


# ------------------------------------------------------------------ contrast
def contrastive_loss(alpha_a: torch.Tensor, alpha_p: torch.Tensor,
                     kernel: Kernel, temperature: float = 0.05,
                     hard_negatives: torch.Tensor | None = None,
                     symmetric: bool = True) -> torch.Tensor:
    """InfoNCE over the kernel, with in-batch and optional explicit negatives.

    Row i of ``alpha_p`` is the positive for row i of ``alpha_a``; every other
    row of ``alpha_p`` is an in-batch negative. ``hard_negatives`` (m, N) are
    appended as negatives shared across the batch -- that is where the role
    reversals, negations and quantity mismatches go, and the brief is right that
    they matter far more than unrelated topics. Topic negatives are separated by
    almost any embedding; those three are what the mixture is at risk of
    conflating.

    Note the kernels here are bounded (Hellinger in [0, 1], the normalised
    learned kernel likewise), unlike the unbounded cosine logits InfoNCE is
    usually written for. ``temperature`` therefore sets the entire usable logit
    range -- at T = 0.05 the span is [0, 20] -- so it is a primary
    hyperparameter, not a detail to leave at a copied default.
    """
    candidates = alpha_p if hard_negatives is None else \
        torch.cat([alpha_p, hard_negatives], dim=0)

    logits = kernel(alpha_a, candidates) / temperature
    labels = torch.arange(alpha_a.shape[0], device=alpha_a.device)
    loss = F.cross_entropy(logits, labels)

    if symmetric:
        # The reverse direction, positives-as-anchors. Hard negatives stay
        # attached to the anchors they were mined for, so they are not reused
        # here -- only the in-batch square is symmetric.
        rev = kernel(alpha_p, alpha_a) / temperature
        loss = 0.5 * (loss + F.cross_entropy(rev, labels))
    return loss


# ------------------------------------------------------------- reconstruction
def reconstruction_loss(y_hat: torch.Tensor, y_teacher: torch.Tensor,
                        normalize_teacher: bool = True,
                        mode: str = "cosine") -> torch.Tensor:
    """Reconstruct the teacher embedding from the mixture, scoring direction.

    ``mode="cosine"`` is the default, and the choice is load-bearing rather than
    stylistic. It was made from measurement -- see
    scripts/decoder_geometry.py and results/ceiling/decoder_geometry_n256.json.

    WHY NOT SQUARED ERROR
    ---------------------
    Under ``mode="l2"`` against L2-normalised targets, the fitted archetypes at
    N=256 came out with a median norm of **17.6x the data radius**, and the
    nearest real text to an archetype had cosine 0.45 out of 50,000 candidates.
    That is fatal to the point of the model: column d_i is supposed to be the
    teacher-space meaning of primitive i, and a point no text resembles cannot
    serve as one. The mixture was reconstructing by cancellation between distant
    extremes rather than by combining recognisable concepts.

    The cause is arithmetic. A convex combination of unit vectors has norm <= 1,
    with equality only when the mass sits on coincident vertices, while the
    targets have norm exactly 1. Squared error therefore charges the model for a
    norm that an interpretable, data-adjacent set of archetypes structurally
    cannot reach, and the only way to pay is to push the archetypes outward. The
    uninterpretable decoder was not a fitting artefact; it was what the objective
    asked for.

    Measured at N=256, three fits, same data:

        free (unconstrained D)      cosine 0.9879   archetype radius 17.59
        D pinned to unit sphere     cosine 0.7629   archetype radius  1.00
        sphere + free scale         cosine 0.9892   archetype radius  1.00

    Freeing the scale -- which is identical to scoring cosine, since minimising
    ||y - s*r||^2 over s is maximising cos(y, r) -- recovers *all* of the lost
    direction quality and slightly beats the unconstrained fit, with archetypes
    sitting on the data's own sphere. Interpretable archetypes are free; the
    squared-error formulation was simply paying for the wrong thing.

    ``mode="l2"`` is kept so the original formulation can be reproduced, not
    because it is a reasonable alternative.
    """
    if normalize_teacher:
        y_teacher = F.normalize(y_teacher, dim=-1)
    if mode == "cosine":
        return (1.0 - F.cosine_similarity(y_hat, y_teacher, dim=-1)).mean()
    if mode == "l2":
        return ((y_hat - y_teacher) ** 2).sum(dim=-1).mean()
    raise ValueError(f"unknown reconstruction mode {mode!r}; "
                     f"expected 'cosine' or 'l2'")


# ------------------------------------------------------------------ sparsity
def entropy(alpha: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Shannon entropy, safe at exact zeros (0 log 0 := 0)."""
    p = alpha.clamp_min(0)
    return -(p * torch.log(p + _EPS)).sum(dim=dim)


def sparsity_loss(alpha: torch.Tensor) -> torch.Tensor:
    """Concentrate each text's mass on fewer primitives.

    An L1 penalty is useless here and it is worth saying why in code rather than
    finding out in a sweep: ||alpha||_1 = 1 identically on the simplex, so the
    penalty is a constant and its gradient is zero. Entropy is the analogue that
    actually varies.

    Push this too hard and the model becomes a near one-hot classifier, which
    throws away the mixture semantics the whole design is for. Keep lam_s small
    and prefer raising the entmax alpha, which produces exact zeros without
    flattening the surviving mixture.
    """
    return entropy(alpha).mean()


def diversity_loss(alpha: torch.Tensor) -> torch.Tensor:
    """Negative entropy of the batch mean -- minimise to spread usage across N.

    Pairs with ``sparsity_loss``: per-example entropy down (each text is a sparse
    mixture) while batch-mean entropy is up (collectively the batch touches many
    primitives). Optimising only the first is exactly how primitive collapse
    happens.

    Correcting a diagnosed collapse, not a default -- see the module docstring.
    """
    return -entropy(alpha.mean(dim=0))


# ---------------------------------------------------------------- grounding
def ground_loss(alpha: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
    """Pin selected coordinates using labelled multilingual concept examples.

    ``targets`` are target distributions over primitives (n, N) -- a one-hot row
    where a text denotes a single primitive, as the brief puts it, "encourage its
    distribution toward the corresponding simplex vertex". ``mask`` (n,) selects
    which rows carry supervision, since most of a batch will not.

    Cross-entropy rather than L2 because the targets are distributions and the
    prediction is one; it also keeps the gradient well-behaved as alpha
    approaches a vertex.
    """
    ce = -(targets * torch.log(alpha.clamp_min(_EPS))).sum(dim=-1)
    if mask is None:
        return ce.mean()
    mask = mask.to(ce.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (ce * mask).sum() / denom


# -------------------------------------------------------------- diagnostics
@torch.no_grad()
def dead_coordinates(alpha: torch.Tensor, threshold: float = 1e-3) -> int:
    """Coordinates no text in this batch activates above ``threshold``.

    The number to watch. It should be reported at every stage of the alpha
    annealing schedule, because sparsemax-like projections kill coordinates
    silently: nothing in the loss goes up when a primitive dies.
    """
    return int((alpha.max(dim=0).values < threshold).sum().item())


@torch.no_grad()
def usage_stats(alpha: torch.Tensor, threshold: float = 1e-3) -> dict:
    """Per-batch summary of how the inventory is being used."""
    active = (alpha > threshold).float()
    return {
        "dead": dead_coordinates(alpha, threshold),
        "mean_active_per_text": active.sum(dim=-1).mean().item(),
        "mean_entropy": entropy(alpha).mean().item(),
        "batch_mean_entropy": entropy(alpha.mean(dim=0)).item(),
        "max_coord_share": alpha.mean(dim=0).max().item(),
    }
