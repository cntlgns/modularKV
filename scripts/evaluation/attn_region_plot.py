"""Plot region-attention comparison across configs from attn_region_compare.py.

Reads <in_dir>/<dataset>__<config>.json for the configs given by --configs and
writes, per dataset:

  figs/<dataset>__regions_compare.png
      2 rows (question rows / gen rows) x N configs: stacked region bars by
      layer (sys incl sink / docs / question / rest), the same view as the old
      attn_analysis __regions.png but one column per config instead of per policy.

  figs/<dataset>__docs_compare.png
      docs-region mass per layer, all configs overlaid, question vs gen panels —
      the headline "does KVLink0 recover the oracle's doc attention?" plot.

Default configs (in order): baseline, base_broken, kvlink0_step4000.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Stacked-region colors (sys incl sink / docs / question / rest).
REGION_COLORS = {"sys": "#7f7f7f", "docs": "#1f77b4",
                 "question": "#2ca02c", "rest": "#d9d9d9"}
CONFIG_COLORS = {
    "baseline": "#1f77b4",
    "base_broken": "#d62728",
    "kvlink0_step4000": "#ff7f0e",
}
CONFIG_LABELS = {
    "baseline": "baseline (oracle, full cache)",
    "base_broken": "base @ recover_pos_enc (broken cache)",
    "kvlink0_step4000": "KVLink0 ft-4000 @ recover_pos_enc",
}


def stacked(mean: np.ndarray):
    """mean (n_layers,4)=[sink,sys,docs,question] -> dict of stack components."""
    sys_ = mean[:, 1]
    docs = mean[:, 2]
    ques = mean[:, 3]
    rest = np.clip(1.0 - (sys_ + docs + ques), 0, 1)
    return {"sys": sys_, "docs": docs, "question": ques, "rest": rest}


def load_cfg(in_dir: Path, dataset: str, cfg: str):
    p = in_dir / f"{dataset}__{cfg}.json"
    if not p.is_file():
        return None
    with open(p) as f:
        return json.load(f)


def render(in_dir: Path, figs_dir: Path, dataset: str, configs: List[str]):
    data = {c: load_cfg(in_dir, dataset, c) for c in configs}
    data = {c: d for c, d in data.items() if d is not None}
    if not data:
        print(f"[skip] {dataset}: no config JSONs found")
        return
    present = [c for c in configs if c in data]

    # ---- Figure 1: stacked region bars, rows=question/gen, cols=configs ---- #
    row_keys = [("question_rows", "question rows"), ("gen_rows", "generation rows")]
    fig, axes = plt.subplots(2, len(present), figsize=(4 * len(present), 7),
                             sharey=True, squeeze=False)
    for ri, (rk, rtitle) in enumerate(row_keys):
        for ci, cfg in enumerate(present):
            ax = axes[ri][ci]
            blk = data[cfg].get(rk)
            if blk is None:
                ax.set_title(f"{cfg}\n(no {rtitle})", fontsize=8)
                ax.axis("off")
                continue
            mean = np.asarray(blk["mean"])              # (n_layers, 4)
            comp = stacked(mean)
            x = np.arange(mean.shape[0])
            bot = np.zeros_like(x, dtype=float)
            for name in ["sys", "docs", "question", "rest"]:
                ax.bar(x, comp[name], bottom=bot, width=1.0,
                       color=REGION_COLORS[name], label=name)
                bot = bot + comp[name]
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25, axis="y")
            n = blk.get("n", "?")
            if ri == 0:
                ax.set_title(f"{CONFIG_LABELS.get(cfg, cfg)}", fontsize=9)
            ax.set_xlabel("layer")
            if ci == 0:
                ax.set_ylabel(f"{rtitle}\nattention mass")
            ax.text(0.02, 0.97, f"n={n}", transform=ax.transAxes,
                    fontsize=7, va="top", ha="left")
    axes[0][-1].legend(fontsize=7, loc="upper right", ncol=2)
    fig.suptitle(f"{dataset}: region attention by layer "
                 f"(question vs generation rows)", fontsize=11)
    fig.tight_layout()
    out1 = figs_dir / f"{dataset}__regions_compare.png"
    fig.savefig(out1, dpi=120)
    plt.close(fig)

    # ---- Figure 2: docs-region mass overlay, question vs gen ---- #
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, (rk, rtitle) in zip(axes, row_keys):
        for cfg in present:
            blk = data[cfg].get(rk)
            if blk is None:
                continue
            mean = np.asarray(blk["mean"])
            std = np.asarray(blk["std"])
            docs = mean[:, 2]
            ds = std[:, 2]
            x = np.arange(mean.shape[0])
            color = CONFIG_COLORS.get(cfg)
            ax.plot(x, docs, marker="o", ms=3, linewidth=1.5, color=color,
                    label=CONFIG_LABELS.get(cfg, cfg))
            ax.fill_between(x, docs - ds, docs + ds, alpha=0.10, color=color)
        ax.set_title(rtitle)
        ax.set_xlabel("layer")
        ax.set_ylabel("attention mass on docs")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"{dataset}: docs-region attention — oracle vs broken vs KVLink0",
                 fontsize=11)
    fig.tight_layout()
    out2 = figs_dir / f"{dataset}__docs_compare.png"
    fig.savefig(out2, dpi=120)
    plt.close(fig)

    # ---- console summary ---- #
    print(f"\n===== {dataset} =====")
    for rk, rtitle in row_keys:
        print(f"  [{rtitle}] mean-over-layers region mass:")
        for cfg in present:
            blk = data[cfg].get(rk)
            if blk is None:
                print(f"    {cfg:<28} (none)")
                continue
            mean = np.asarray(blk["mean"])
            comp = stacked(mean)
            mo = {k: float(np.mean(v)) for k, v in comp.items()}
            print(f"    {cfg:<28} sys={mo['sys']:.3f} docs={mo['docs']:.3f} "
                  f"question={mo['question']:.3f} rest={mo['rest']:.3f}")
    print(f"  -> {out1.name}, {out2.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="result/attn_region_compare")
    ap.add_argument("--datasets", nargs="+",
                    default=["nq", "hqa_gold", "wiki", "musique", "block_qa"])
    ap.add_argument("--configs", nargs="+",
                    default=["baseline", "base_broken", "kvlink0_step4000"])
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    figs_dir = in_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        render(in_dir, figs_dir, ds, args.configs)
    print(f"\n[figs] written to {figs_dir}/")


if __name__ == "__main__":
    main()
