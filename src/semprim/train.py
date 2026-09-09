"""Stage B: train the bottleneck head over cached embeddings.

    python -m semprim.train --config configs/smoke.yaml

Every batch is a lookup into the fp16 cache written by encode.py, so a run costs
minutes and the whole sweep costs an afternoon. See encode.py for why that is the
load-bearing decision in this project.

The config file is the record of what ran: every weight, the kernel, the alpha
schedule and the seed are read from it and copied verbatim into the results
directory. A run whose settings live in edited source is a run nobody can
reproduce.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import yaml

from . import losses as L
from .archetypal import archetypal_fit, fit_quality
from .corpus import assert_disjoint, load_corpus, split_pairs
from .encode import ensure_cache
from .kernels import build_kernel
from .model import PrimitiveModel


# --------------------------------------------------------------------- setup
def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(requested: str | None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def alpha_for_epoch(schedule, epoch: int) -> float:
    """Entmax alpha from a [[epoch, alpha], ...] schedule.

    Annealing rather than a fixed alpha because a projection with exact zeros
    has no gradient outside its support: a primitive that dies in epoch 0 is
    dead for the whole run. Starting at softmax (alpha = 1, dense) keeps every
    coordinate receiving gradient while the representation forms, and tightening
    afterwards produces the sparsity without the dead units. model.py has the
    longer version of this argument.
    """
    current = 1.0
    for start, value in sorted(schedule, key=lambda p: p[0]):
        if epoch >= start:
            current = float(value)
    return current


@dataclass
class Batch:
    h_a: torch.Tensor
    h_b: torch.Tensor
    y_a: torch.Tensor


def build_texts(cfg):
    """Return (texts, pair_index) for the configured source.

    ``pair_index`` is a list of (i, j) into ``texts``: i and j are meaning
    equivalent, and are what L_align pulls together.

    If ``scripts/stage_a.py`` has already written a ``corpus.json`` beside the
    cache, it is reused verbatim. That is not only a speed matter: re-streaming
    OPUS-100 per run costs about a minute, needs the network, and -- because the
    datasets cache defaults to the user profile rather than HF_HOME unless the
    variable is set -- quietly writes to the system drive, which on this machine
    has under 7 GB free. Reusing the file also guarantees every N in a sweep
    trains on exactly the same corpus, which re-streaming does not: the loader
    takes the first N rows of a shard after filtering, so a change in the
    dataset upstream would silently give two sweep points different data.
    """
    source = cfg["data"]["source"]

    corpus_path = os.path.join(cfg.get("cache_dir", ""), "corpus.json")
    if source != "synthetic" and os.path.exists(corpus_path):
        corpus = load_corpus(cfg["cache_dir"])
        # TRAIN PAIRS ONLY. This previously returned every pair, held-out
        # included, so evaluation scored data the model had trained on -- see
        # semprim/corpus.py for what that cost.
        pairs, held = split_pairs(corpus)
        assert_disjoint(pairs, held)
        print(f"reusing corpus.json from {corpus_path}: "
              f"{len(pairs):,} train pairs, {len(held):,} held out")
        return corpus["texts"], pairs

    if source == "synthetic":
        from .data import synthetic_pairs
        rows = synthetic_pairs(n=cfg["data"]["n"],
                               n_topics=cfg["data"].get("n_topics", 16),
                               seed=cfg.get("seed", 0))
        texts, pairs = [], []
        for a, b, _topic in rows:
            pairs.append((len(texts), len(texts) + 1))
            texts.extend([a, b])
        return texts, pairs
    if source == "opus":
        from .data import load_translation_pairs
        rows = load_translation_pairs(
            limit_per_lang=cfg["data"].get("limit_per_lang", 20000))
        texts, pairs = [], []
        for a, b, _lang in rows:
            pairs.append((len(texts), len(texts) + 1))
            texts.extend([a, b])
        return texts, pairs
    raise ValueError(f"unknown data source {source!r}")


# ------------------------------------------------------------------ training
def train(cfg: dict, out_dir: str) -> dict:
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(cfg.get("device"))

    texts, pairs = build_texts(cfg)
    print(f"corpus: {len(texts):,} texts, {len(pairs):,} equivalence pairs")

    cache = ensure_cache(texts, cfg["cache_dir"],
                         model_name=cfg.get("encoder", "sentence-transformers/LaBSE"),
                         batch_size=cfg.get("encode_batch_size", 64),
                         max_seq_length=cfg.get("max_seq_length", 64))
    H = torch.from_numpy(np.asarray(cache, dtype=np.float32))
    print(f"cache: {tuple(H.shape)}")

    pair_idx = torch.tensor(pairs, dtype=torch.long)
    n_primitives = int(cfg["model"]["n_primitives"])

    # The teacher is the frozen encoder itself, so reconstruction is
    # autoencoding through the simplex -- i.e. archetypal analysis. Fit the
    # non-amortised version first: it both seeds D and gives the error the
    # trained head is measured against.
    recon_mode = cfg["loss"].get("reconstruct_mode", "cosine")
    ceiling = None
    if cfg.get("archetypal", {}).get("enabled", True):
        sample = H[torch.randperm(H.shape[0])[:cfg.get("archetypal", {}).get("sample", 5000)]]
        t0 = time.perf_counter()
        # free_scale must track reconstruct_mode. A cosine loss is invariant to
        # the reconstruction's magnitude, which is exactly what a free
        # per-example scale expresses -- so fitting the ceiling without it would
        # bound a different objective than the head optimises, and would not
        # bound the head at all.
        sample_n = torch.nn.functional.normalize(sample.to(device), dim=-1)
        D0, A0, _ = archetypal_fit(
            sample_n, n_primitives,
            outer=cfg.get("archetypal", {}).get("outer", 15),
            inner=cfg.get("archetypal", {}).get("inner", 50), seed=seed,
            normalize=False, free_scale=(recon_mode == "cosine"))
        quality = fit_quality(D0, A0, sample_n)
        # Reported in the loss's own units, so the comparison at the end is a
        # comparison and not a units error.
        ceiling = quality["cosine_loss"] if recon_mode == "cosine" \
            else quality["mse"]
        print(f"archetypal ceiling at N={n_primitives} "
              f"({recon_mode}): {ceiling:.6f}   "
              f"[mse {quality['mse']:.6f}, cos {quality['cosine']:.4f}] "
              f"({time.perf_counter() - t0:.1f}s)")
    else:
        D0 = None

    model = PrimitiveModel(
        in_dim=H.shape[1], teacher_dim=H.shape[1],
        hidden=int(cfg["model"].get("hidden", 1024)),
        n_primitives=n_primitives,
        alpha=alpha_for_epoch(cfg["train"].get("alpha_schedule", [[0, 1.0]]), 0),
        tau_init=float(cfg["model"].get("tau_init", 1.0)),
        learn_tau=bool(cfg["model"].get("learn_tau", True)),
        dropout=float(cfg["model"].get("dropout", 0.0)),
    ).to(device)
    if D0 is not None:
        model.init_decoder_(D0.to(device))

    kernel = build_kernel(cfg["kernel"]["name"], n_primitives,
                          **{k: v for k, v in cfg["kernel"].items()
                             if k != "name"}).to(device)

    params = list(model.parameters()) + list(kernel.parameters())
    opt = torch.optim.AdamW(params, lr=float(cfg["train"]["lr"]),
                            weight_decay=float(cfg["train"].get("weight_decay", 0.0)))

    w = cfg["loss"]
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    temperature = float(cfg["train"].get("temperature", 0.05))
    geometry = w.get("align_geometry", "l2")

    H = H.to(device)
    history = []

    for epoch in range(epochs):
        model.bottleneck.set_alpha(
            alpha_for_epoch(cfg["train"].get("alpha_schedule", [[0, 1.0]]), epoch))
        perm = torch.randperm(pair_idx.shape[0], device=device)
        epoch_stats, n_batches = {}, 0

        for start in range(0, perm.shape[0], batch_size):
            sel = pair_idx[perm[start:start + batch_size].cpu()]
            if sel.shape[0] < 2:
                continue                       # InfoNCE needs a negative
            h_a = H[sel[:, 0].to(device)]
            h_b = H[sel[:, 1].to(device)]

            alpha_a = model(h_a)
            alpha_b = model(h_b)

            loss = h_a.new_zeros(())
            parts = {}
            if w.get("contrast", 0):
                c = L.contrastive_loss(alpha_a, alpha_b, kernel, temperature)
                loss = loss + float(w["contrast"]) * c
                parts["contrast"] = c.item()
            if w.get("align", 0):
                a = L.align_loss(alpha_a, alpha_b, geometry=geometry)
                loss = loss + float(w["align"]) * a
                parts["align"] = a.item()
            if w.get("reconstruct", 0):
                r = L.reconstruction_loss(
                    model.reconstruct(alpha_a), h_a,
                    mode=w.get("reconstruct_mode", "cosine"))
                loss = loss + float(w["reconstruct"]) * r
                parts["reconstruct"] = r.item()
            if w.get("sparsity", 0):
                s = L.sparsity_loss(alpha_a)
                loss = loss + float(w["sparsity"]) * s
                parts["sparsity"] = s.item()
            if w.get("diversity", 0):
                d = L.diversity_loss(alpha_a)
                loss = loss + float(w["diversity"]) * d
                parts["diversity"] = d.item()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["train"].get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(params, float(cfg["train"]["grad_clip"]))
            opt.step()

            parts["total"] = loss.item()
            for k, v in parts.items():
                epoch_stats[k] = epoch_stats.get(k, 0.0) + v
            n_batches += 1

        with torch.no_grad():
            alpha_all = model(H[:min(4096, H.shape[0])])
            usage = L.usage_stats(alpha_all)
        row = {k: v / max(n_batches, 1) for k, v in epoch_stats.items()}
        row.update({"epoch": epoch, "alpha": model.bottleneck.alpha,
                    "tau": model.bottleneck.tau.item(), **usage})
        history.append(row)
        print(f"epoch {epoch:>3}  alpha={row['alpha']:.2f}  "
              f"total={row.get('total', 0):.4f}  "
              f"dead={usage['dead']}/{n_primitives}  "
              f"active/text={usage['mean_active_per_text']:.1f}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "kernel": kernel.state_dict()},
               os.path.join(out_dir, "checkpoint.pt"))
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    summary = {
        "history": history,
        "archetypal_ceiling": ceiling,
        "reconstruct_mode": recon_mode,
        "final": history[-1] if history else None,
        "n_primitives": n_primitives,
        "device": str(device),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # SPARSITY-MATCHED CEILING -- computed after training, because it needs the
    # head's observed support size.
    #
    # The dense ceiling above is NOT a bound on a sparse head, and treating it
    # as one gives the wrong answer with confidence. At N=256 the head runs
    # entmax alpha=2 and activates ~21 of 256 coordinates; against the dense fit
    # it read "29.5x above ceiling", implying it could not compute the mapping.
    # Matched to k=21 the same head is 1.84x above, and at N=64 it is 1.33x --
    # a head with real headroom, not a broken one. The dense figure is kept for
    # continuity with Milestone 2 but must not be read as a bound.
    matched = None
    if history and cfg.get("archetypal", {}).get("enabled", True):
        k = int(round(history[-1].get("mean_active_per_text") or 0))
        if 0 < k < n_primitives:
            Dk, Ak, _ = archetypal_fit(
                sample_n, n_primitives,
                outer=cfg.get("archetypal", {}).get("outer", 15),
                inner=cfg.get("archetypal", {}).get("inner", 50), seed=seed,
                normalize=False, topk=k)
            qk = fit_quality(Dk, Ak, sample_n)
            matched = {"k": k,
                       "ceiling": qk["cosine_loss"] if recon_mode == "cosine"
                       else qk["mse"]}
    summary["sparsity_matched_ceiling"] = matched

    if history:
        got = history[-1].get("reconstruct")
        if got is not None and matched is not None:
            mc = matched["ceiling"]
            print(f"\nreconstruction: head {got:.6f}")
            print(f"  vs k-sparse ceiling (k={matched['k']}): {mc:.6f} "
                  f"-> {got / max(mc, 1e-12):.2f}x   [the fair comparison]")
            if ceiling is not None:
                print(f"  vs dense ceiling (all {n_primitives}): "
                      f"{ceiling:.6f} -> {got / max(ceiling, 1e-12):.2f}x   "
                      f"[not a bound on a sparse head]")
            if got < mc * 0.999:
                # The head cannot legitimately beat a per-example fit at the
                # same N and the same support size -- that optimum is over a
                # strict superset of what the head can express. Beating it means
                # the two are not being measured on the same thing.
                print("  WARNING: head beat the matched ceiling. The two are "
                      "not comparable -- check normalisation, N, k, and that "
                      "both use the same targets.")
        elif got is not None and ceiling is not None:
            print(f"\nreconstruction: head {got:.6f} vs dense ceiling "
                  f"{ceiling:.6f} (ratio {got / max(ceiling, 1e-12):.2f}x); "
                  f"no sparsity-matched ceiling computed")

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the simplex bottleneck.")
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None, help="results directory")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    out = a.out or os.path.join("results", cfg.get("run_name", "run"))
    summary = train(cfg, out)
    print(f"\nwrote {out}")
    return summary


if __name__ == "__main__":
    main()
