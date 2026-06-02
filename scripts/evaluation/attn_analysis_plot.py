"""Aggregate + plot attention-analysis outputs from attn_analysis.py.

Reads attn_analysis_results/<dataset>__<model>.json files and writes:
  - figs/<dataset>__<model>__entropy.png        : entropy + entropy_excl_first
  - figs/<dataset>__<model>__sink.png           : first-token mass per policy
  - figs/<dataset>__<model>__docs.png           : total docs mass per policy
  - figs/<dataset>__<model>__regions.png        : stacked region bars
  - figs/<dataset>__<model>__per_gold.png       : (hqa_gold) per-doc mass curves
  - figs/<dataset>__<model>__gold_vs_dist.png   : (others) gold vs distractor-mean
  - tables: prints across-policy summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCALAR_FIELDS = ["mass_sys", "mass_qprev", "mass_first",
                 "mass_docs_total", "entropy", "entropy_excl_first"]

POLICY_COLORS = {
    "baseline": "#1f77b4",
    "recover_pos_enc": "#ff7f0e",
    "modular": "#2ca02c",
    "recover_cross_attn_oracle_pos": "#d62728",
}


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def scalar_arrays(d):
    """Return per-policy (mean, std) ndarrays of shape (n_layers, n_fields)."""
    res = d["results"]
    mean, std = {}, {}
    for p, samples in res.items():
        if not samples:
            continue
        n_samp = len(samples)
        n_layers = len(samples[0])
        arr = np.zeros((n_samp, n_layers, len(SCALAR_FIELDS)))
        for si, row in enumerate(samples):
            for li, layer in enumerate(row):
                for fi, k in enumerate(SCALAR_FIELDS):
                    arr[si, li, fi] = layer[k]
        mean[p] = arr.mean(axis=0)
        std[p]  = arr.std(axis=0)
    return mean, std


def gold_vs_dist_arrays(d):
    """For each policy, compute mean over samples of:
      gold_mass[layer] = mean over gold docs of per_doc_mass
      dist_mass[layer] = mean over non-gold docs of per_doc_mass
    Returns dict policy -> (gold_mean, gold_std, dist_mean, dist_std) per layer.
    Samples without any non-gold doc contribute nothing to the distractor mean.
    Samples without any gold doc contribute nothing to gold mean.
    """
    gold_idx_list = d["gold_idx_list"]
    out = {}
    for p, samples in d["results"].items():
        n_samp = len(samples)
        n_layers = len(samples[0])
        gold_vals = [[] for _ in range(n_layers)]
        dist_vals = [[] for _ in range(n_layers)]
        for si, row in enumerate(samples):
            gold = set(gold_idx_list[si])
            for li, layer in enumerate(row):
                pdm = layer["mass_per_doc"]
                gv = [m for i, m in enumerate(pdm) if i in gold]
                dv = [m for i, m in enumerate(pdm) if i not in gold]
                if gv:
                    gold_vals[li].append(np.mean(gv))
                if dv:
                    dist_vals[li].append(np.mean(dv))
        gm = np.array([np.mean(v) if v else np.nan for v in gold_vals])
        gs = np.array([np.std(v)  if v else 0.0    for v in gold_vals])
        dm = np.array([np.mean(v) if v else np.nan for v in dist_vals])
        ds = np.array([np.std(v)  if v else 0.0    for v in dist_vals])
        out[p] = (gm, gs, dm, ds)
    return out


def per_gold_doc_arrays(d):
    """For hqa_gold style data (all docs are gold). For each doc-rank k (0,1,...)
    compute mean across samples of per_doc_mass[k] (NaN if any sample lacks rank-k).
    """
    out = {}
    for p, samples in d["results"].items():
        n_samp = len(samples)
        n_layers = len(samples[0])
        # how many docs per sample? Take max so the per-doc list extends
        max_docs = max(len(s[0]["mass_per_doc"]) for s in samples)
        per_doc = np.full((n_layers, max_docs), np.nan)
        count   = np.zeros((n_layers, max_docs), dtype=int)
        acc     = np.zeros((n_layers, max_docs))
        for si, row in enumerate(samples):
            for li, layer in enumerate(row):
                for k, m in enumerate(layer["mass_per_doc"]):
                    acc[li, k]   += m
                    count[li, k] += 1
        with np.errstate(invalid="ignore"):
            per_doc = acc / np.where(count > 0, count, 1)
            per_doc[count == 0] = np.nan
        out[p] = per_doc  # (n_layers, max_docs)
    return out


def plot_curve(ax, mean, std, label, color=None, linestyle="-", marker="o"):
    L = mean.shape[0]
    x = np.arange(L)
    ax.plot(x, mean, marker=marker, linestyle=linestyle, label=label, linewidth=1.4, color=color)
    ax.fill_between(x, mean - std, mean + std, alpha=0.14, color=color)


def render(json_path: Path, figs_dir: Path):
    d = load_json(json_path)
    model = d.get("model", "?")
    dataset = d.get("dataset", json_path.stem)
    stem = f"{dataset}__{model.split('/')[-1]}"
    print(f"\n========== {stem} ==========")

    mean, std = scalar_arrays(d)
    policies = list(mean.keys())
    n_layers = next(iter(mean.values())).shape[0]
    n_samp = len(d["samples"])
    fi = {k: i for i, k in enumerate(SCALAR_FIELDS)}

    # Per-layer summary table
    print(f"  {n_samp} samples, {n_layers} layers, policies={policies}")
    print(f"\n  Per-policy mean (avg over layers):")
    print(f"  {'policy':<32} {'sys':>7} {'qprev':>7} {'first':>7} {'docs':>7} {'H':>7} {'H_ef':>7}")
    for p in policies:
        row = mean[p].mean(axis=0)
        print(f"  {p:<32} " + " ".join(f"{row[fi[k]]:7.3f}" for k in SCALAR_FIELDS))

    # mid-layer (33–66% depth) entropy summary
    lo = int(0.33 * n_layers); hi = int(0.66 * n_layers)
    print(f"\n  Mid-layer ({lo}-{hi}) entropy mean (raw vs excl_first):")
    for p in policies:
        e  = mean[p][lo:hi+1, fi["entropy"]].mean()
        ef = mean[p][lo:hi+1, fi["entropy_excl_first"]].mean()
        print(f"    {p:<32}  H={e:.3f}   H_excl_first={ef:.3f}   Δ={e-ef:+.3f}")

    # --- Figure 1: entropy (raw + excl_first, side by side) --- #
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)
    for p in policies:
        plot_curve(axes[0], mean[p][:, fi["entropy"]], std[p][:, fi["entropy"]],
                   p, POLICY_COLORS.get(p))
        plot_curve(axes[1], mean[p][:, fi["entropy_excl_first"]],
                   std[p][:, fi["entropy_excl_first"]],
                   p, POLICY_COLORS.get(p))
    axes[0].set_title("entropy (raw)")
    axes[1].set_title("entropy excluding first token")
    for ax in axes:
        ax.set_xlabel("layer")
        ax.set_ylabel("attention entropy (nats)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{stem}: question-row attention entropy")
    fig.tight_layout()
    fig.savefig(figs_dir / f"{stem}__entropy.png", dpi=120)
    plt.close(fig)

    # --- Figure 2: sink (first-token mass) --- #
    fig, ax = plt.subplots(figsize=(7, 4))
    for p in policies:
        plot_curve(ax, mean[p][:, fi["mass_first"]], std[p][:, fi["mass_first"]],
                   p, POLICY_COLORS.get(p))
    ax.set_xlabel("layer"); ax.set_ylabel("mass on first token (sink)")
    ax.set_title(f"{stem}: sink probe")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figs_dir / f"{stem}__sink.png", dpi=120); plt.close(fig)

    # --- Figure 3: docs total mass --- #
    fig, ax = plt.subplots(figsize=(7, 4))
    for p in policies:
        plot_curve(ax, mean[p][:, fi["mass_docs_total"]], std[p][:, fi["mass_docs_total"]],
                   p, POLICY_COLORS.get(p))
    ax.set_xlabel("layer"); ax.set_ylabel("mass on all docs")
    ax.set_title(f"{stem}: question -> docs (all)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figs_dir / f"{stem}__docs.png", dpi=120); plt.close(fig)

    # --- Figure 4: stacked region bars per policy --- #
    fig, axes = plt.subplots(1, len(policies), figsize=(4 * len(policies), 4), sharey=True)
    if len(policies) == 1:
        axes = [axes]
    for ax, p in zip(axes, policies):
        x = np.arange(n_layers)
        sys_ = mean[p][:, fi["mass_sys"]]
        qp   = mean[p][:, fi["mass_qprev"]]
        dt   = mean[p][:, fi["mass_docs_total"]]
        other = np.clip(1.0 - (sys_ + qp + dt), 0, 1)
        bot = np.zeros_like(x, dtype=float)
        for arr, name in [(sys_, "sys"), (dt, "docs"), (qp, "qprev"), (other, "other")]:
            ax.bar(x, arr, bottom=bot, label=name)
            bot = bot + arr
        ax.set_title(p, fontsize=10); ax.set_xlabel("layer")
        ax.set_ylim(0, 1.05); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("attention mass")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{stem}: question-row mass by region")
    fig.tight_layout()
    fig.savefig(figs_dir / f"{stem}__regions.png", dpi=120); plt.close(fig)

    # --- Figure 5: hqa_gold → per-doc curves; others → gold vs distractor --- #
    if dataset == "hqa_gold":
        per_doc = per_gold_doc_arrays(d)
        fig, axes = plt.subplots(1, len(policies), figsize=(4 * len(policies), 4), sharey=True)
        if len(policies) == 1:
            axes = [axes]
        for ax, p in zip(axes, policies):
            arr = per_doc[p]  # (n_layers, max_docs)
            for k in range(arr.shape[1]):
                ax.plot(np.arange(n_layers), arr[:, k], marker="o", linewidth=1.4, label=f"gold doc #{k}")
            ax.set_title(p, fontsize=10); ax.set_xlabel("layer"); ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        axes[0].set_ylabel("mass on a single gold doc")
        fig.suptitle(f"{stem}: question -> each gold doc (gold-only, all docs are gold)")
        fig.tight_layout()
        fig.savefig(figs_dir / f"{stem}__per_gold.png", dpi=120); plt.close(fig)
    else:
        gd = gold_vs_dist_arrays(d)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for p in policies:
            gm, gs, dm, ds = gd[p]
            x = np.arange(n_layers)
            color = POLICY_COLORS.get(p)
            ax.plot(x, gm, marker="o", linewidth=1.6, color=color, label=f"{p} gold")
            ax.fill_between(x, gm - gs, gm + gs, alpha=0.12, color=color)
            ax.plot(x, dm, marker="x", linestyle="--", linewidth=1.2, color=color,
                    label=f"{p} distractor(mean)")
        ax.set_xlabel("layer"); ax.set_ylabel("mass on a single doc")
        ax.set_title(f"{stem}: gold vs distractor (per-doc avg)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figs_dir / f"{stem}__gold_vs_dist.png", dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="attn_analysis_results")
    ap.add_argument("--glob", default="*.json")
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    figs_dir = in_dir / "figs"; figs_dir.mkdir(exist_ok=True)
    files = sorted(p for p in in_dir.glob(args.glob) if "__" in p.stem)
    if not files:
        print(f"[empty] no JSONs in {in_dir}")
        return
    for path in files:
        try:
            render(path, figs_dir)
        except Exception as e:
            print(f"[err] {path}: {e}")
    print(f"\n[figs] written to {figs_dir}/")


if __name__ == "__main__":
    main()
