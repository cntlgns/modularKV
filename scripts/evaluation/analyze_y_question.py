"""Diagnose why y_question is harder to predict than y_sink / y_sys / y_docs.

Runs three orthogonal checks on the heldout block_qa-test parquets:

1. **Per-target marginal variance** — total Var(y), Var(y | layer, head),
   intra-(L,H) Var(y) averaged over cells. Compares how much variance is
   between cells vs within cells. If y_question is dominated by within-cell
   variance, that's variance the regressor can never explain with cell-level
   features (layer, head).

2. **Univariate Pearson + Spearman correlation** between each cont. feature
   and each target. Ranks features by |corr| for y_question vs others —
   reveals whether the available features carry less signal for question.

3. **GBM permutation importance** on a small sub-sample (default 500K rows)
   for each saved model. Confirms which features the trained model relies on.

Outputs:
    analyze_y_question/
       summary.json        — all numbers
       y_question_perLH_mean.png  — heatmap of mean(y_question) per (layer, head)
       y_question_perLH_std.png   — heatmap of std(y_question) per (layer, head)
       perLH_mean_grid.png        — same heatmaps for all 4 targets side-by-side

Usage:
    source .venv/bin/activate
    python scripts/evaluation/analyze_y_question.py \
        --in_dir analysis/attention_score_analysis/attn_raw_gen_results \
        --gbm_dir analysis/attention_score_analysis/attn_score_gen_models \
        --out_dir analysis/attention_score_analysis/analyze_y_question
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGETS = ["y_sink", "y_sys", "y_docs", "y_question"]


def stream_load(in_dir: Path, glob_pattern: str, max_rows: int, seed: int = 0,
                batch_size: int = 200_000) -> pd.DataFrame:
    """Same pyarrow-streaming loader as the trainer."""
    import pyarrow.parquet as pq
    files = sorted(in_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No parquet under {in_dir} matching {glob_pattern!r}")
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    keep_frac = max_rows / total if max_rows and max_rows > 0 and total > max_rows else None
    print(f"  [load] total_rows={total:,}  keep_frac={keep_frac}", flush=True)
    rng = np.random.default_rng(seed)
    parts = []
    for f in files:
        pq_file = pq.ParquetFile(f)
        for batch in pq_file.iter_batches(batch_size=batch_size):
            d = batch.to_pandas()
            if keep_frac is not None:
                d = d.sample(frac=keep_frac, random_state=int(rng.integers(0, 2**31 - 1)))
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    for t in TARGETS:
        df[t] = df[t].clip(0.0, 1.0)
    # Derived features (mirror trainer's load()).
    df["prompt_len"] = (df["sys_len"].astype(np.int32) +
                        df["docs_total_len"].astype(np.int32) +
                        df["q_len"].astype(np.int32))
    df["cur_pos"] = df["prompt_len"] + df["gi"].astype(np.int32)
    df["q_to_prompt_ratio"] = df["q_len"].astype(np.float32) / df["prompt_len"].clip(lower=1).astype(np.float32)
    df["q_to_docs_ratio"]   = df["q_len"].astype(np.float32) / df["docs_total_len"].clip(lower=1).astype(np.float32)
    df["gi_to_q_len"]       = df["gi"].astype(np.float32) / df["q_len"].clip(lower=1).astype(np.float32)
    return df


def variance_breakdown(df: pd.DataFrame) -> dict:
    """For each target compute (total var, between-(L,H) var, within-(L,H) avg var,
    cell-mean range). Helps see if signal is mostly per-cell or per-row."""
    out = {}
    for t in TARGETS:
        y = df[t].to_numpy()
        total_var = float(np.var(y))
        # Per-(L, H) cell stats
        grp = df.groupby(["layer", "head"])[t]
        cell_mean = grp.mean()
        cell_var  = grp.var()
        between_var = float(np.var(cell_mean.to_numpy()))
        within_var  = float(np.mean(cell_var.to_numpy()))
        out[t] = {
            "total_var":       total_var,
            "between_LH_var":  between_var,
            "within_LH_var_mean": within_var,
            "between_frac":    float(between_var / max(total_var, 1e-12)),
            "within_frac":     float(within_var / max(total_var, 1e-12)),
            "y_mean":  float(y.mean()),
            "y_std":   float(y.std()),
            "y_p05":   float(np.percentile(y, 5)),
            "y_p50":   float(np.percentile(y, 50)),
            "y_p95":   float(np.percentile(y, 95)),
            "cell_mean_min": float(cell_mean.min()),
            "cell_mean_max": float(cell_mean.max()),
        }
    return out


def univariate_corr(df: pd.DataFrame, feats: list[str]) -> dict:
    """Per-feature Pearson + Spearman with each target. Spearman robust to
    nonlinearity; comparing the two flags monotone-but-nonlinear relations."""
    out = {}
    n = min(len(df), 1_000_000)  # cap for spearman speed
    sub = df.sample(n=n, random_state=0) if len(df) > n else df
    for t in TARGETS:
        out[t] = {}
        y = sub[t].to_numpy()
        for f in feats:
            x = sub[f].to_numpy()
            # Pearson (no scipy dep; compute manually with float64).
            xm = x - x.mean(); ym = y - y.mean()
            denom = np.sqrt((xm * xm).sum() * (ym * ym).sum()) + 1e-12
            pearson = float((xm * ym).sum() / denom)
            # Spearman via rank Pearson
            xr = pd.Series(x).rank().to_numpy()
            yr = pd.Series(y).rank().to_numpy()
            xrm = xr - xr.mean(); yrm = yr - yr.mean()
            sdenom = np.sqrt((xrm * xrm).sum() * (yrm * yrm).sum()) + 1e-12
            spearman = float((xrm * yrm).sum() / sdenom)
            out[t][f] = {"pearson": pearson, "spearman": spearman}
    return out


def perm_importance(df: pd.DataFrame, gbm_dir: Path, n_repeats: int = 3,
                    sub_n: int = 500_000) -> dict:
    """Permutation importance for each saved GBM. R² drop when each feature
    is randomly shuffled — higher = model relies more on that feature."""
    from sklearn.inspection import permutation_importance
    out = {}
    n = min(len(df), sub_n)
    sub = df.sample(n=n, random_state=0) if len(df) > n else df
    for t in TARGETS:
        stem = t.replace("y_", "")
        path = gbm_dir / f"{stem}.joblib"
        if not path.exists():
            print(f"  [warn] no GBM at {path} — skipping {t}")
            continue
        pack = joblib.load(path)
        m = pack["model"]
        cont = pack["cont_feats"]; cat = pack["cat_feats"]
        feats = cont + cat
        X = sub[feats].to_numpy()
        y = sub[t].to_numpy()
        t0 = time.time()
        res = permutation_importance(m, X, y, n_repeats=n_repeats,
                                     random_state=0, scoring="r2", n_jobs=-1)
        out[t] = {
            "feat_order": feats,
            "importance_mean": [float(v) for v in res.importances_mean],
            "importance_std":  [float(v) for v in res.importances_std],
            "n_rows":   int(len(sub)),
            "n_repeats": n_repeats,
            "elapsed_s": round(time.time() - t0, 1),
        }
        print(f"  [perm] {t}: {res.importances_mean.shape[0]} feats, "
              f"top3={list(np.array(feats)[np.argsort(-res.importances_mean)[:3]])}  "
              f"[{out[t]['elapsed_s']}s]")
    return out


def plot_per_LH(df: pd.DataFrame, out_dir: Path, n_layers: int, n_heads: int):
    """4-panel grid: per-(layer, head) mean(y) heatmap for each target."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for i, t in enumerate(TARGETS):
        grp = df.groupby(["layer", "head"])[t]
        m = grp.mean().unstack().reindex(index=range(n_layers), columns=range(n_heads)).to_numpy()
        s = grp.std().unstack().reindex(index=range(n_layers), columns=range(n_heads)).to_numpy()
        # mean
        ax = axes[0, i]
        im = ax.imshow(m, aspect="auto", cmap="viridis", vmin=0,
                       vmax=max(0.05, float(np.nanpercentile(m, 99))))
        ax.set_title(f"mean({t})\n[{float(m.mean()):.3f}, range {float(np.nanmin(m)):.3f}..{float(np.nanmax(m)):.3f}]")
        ax.set_xlabel("head"); ax.set_ylabel("layer")
        plt.colorbar(im, ax=ax)
        # std
        ax = axes[1, i]
        im = ax.imshow(s, aspect="auto", cmap="magma",
                       vmin=0, vmax=max(0.05, float(np.nanpercentile(s, 99))))
        ax.set_title(f"std({t})  (per-cell within-variance)")
        ax.set_xlabel("head"); ax.set_ylabel("layer")
        plt.colorbar(im, ax=ax)
    fig.suptitle("Per-(layer, head) target distribution (heldout block_qa-test sample)", fontsize=14)
    fig.tight_layout()
    p = out_dir / "perLH_mean_std_grid.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"  [plot] wrote {p.name}")


def plot_y_question_vs_gi(df: pd.DataFrame, out_dir: Path,
                          n_layers: int, n_heads: int):
    """Average y_question across (layer, head) cells, plotted vs gi.
    Highlights early vs late generation dynamics."""
    bins = np.arange(0, df["gi"].max() + 1)
    grp = df.groupby("gi")["y_question"]
    mean = grp.mean()
    p10 = grp.quantile(0.1)
    p90 = grp.quantile(0.9)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mean.index, mean.values, lw=2, label="mean", color="C0")
    ax.fill_between(mean.index, p10.values, p90.values, alpha=0.25, color="C0",
                    label="p10–p90")
    ax.set_xlabel("gi (generation step)")
    ax.set_ylabel("y_question (head/layer-averaged)")
    ax.set_title("Attention to question span as generation progresses")
    ax.legend()
    ax.grid(alpha=0.3)
    p = out_dir / "y_question_vs_gi.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    print(f"  [plot] wrote {p.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",
                    default="analysis/attention_score_analysis/attn_raw_gen_results")
    ap.add_argument("--gbm_dir",
                    default="analysis/attention_score_analysis/attn_score_gen_models")
    ap.add_argument("--out_dir",
                    default="analysis/attention_score_analysis/analyze_y_question")
    ap.add_argument("--heldout_glob", default="block_qa_test__*.parquet")
    ap.add_argument("--max_rows", type=int, default=20_000_000,
                    help="Streaming row cap (default 20M, ~22% of heldout).")
    ap.add_argument("--perm_n", type=int, default=500_000,
                    help="Rows used for permutation importance.")
    ap.add_argument("--perm_repeats", type=int, default=3)
    ap.add_argument("--n_layers", type=int, default=32)
    ap.add_argument("--n_heads",  type=int, default=32)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = Path(args.in_dir); gbm_dir = Path(args.gbm_dir)

    print(f"[load] heldout from {in_dir}  glob={args.heldout_glob!r}")
    t0 = time.time()
    df = stream_load(in_dir, args.heldout_glob, max_rows=args.max_rows)
    print(f"[load] rows={len(df):,}  unique samples={df['sample_id'].nunique():,}  "
          f"[{time.time() - t0:.0f}s]", flush=True)

    # All features (raw + derived).
    cont_feats = [
        "gi", "q_len", "n_docs", "docs_total_len", "mean_doc_len", "sys_len",
        "prompt_len", "cur_pos", "q_to_prompt_ratio", "q_to_docs_ratio", "gi_to_q_len",
    ]

    print("[1/4] variance breakdown")
    vb = variance_breakdown(df)
    for t, v in vb.items():
        print(f"  {t:<11} total={v['total_var']:.4e}  "
              f"between_LH={v['between_frac']*100:.1f}%  within_LH={v['within_frac']*100:.1f}%  "
              f"cell_mean range=[{v['cell_mean_min']:.3f}, {v['cell_mean_max']:.3f}]")

    print("[2/4] univariate correlations (sample 1M rows)")
    corr = univariate_corr(df, cont_feats)
    # Print sorted by |spearman| for each target — readable comparison.
    for t in TARGETS:
        feats_sorted = sorted(corr[t].items(), key=lambda kv: -abs(kv[1]["spearman"]))
        top = [(f, c["pearson"], c["spearman"]) for f, c in feats_sorted[:6]]
        print(f"  {t:<11} top-6 |spearman|:")
        for f, p, s in top:
            print(f"      {f:<22} pearson={p:+.3f}  spearman={s:+.3f}")

    print(f"[3/4] permutation importance from saved GBMs in {gbm_dir}")
    if not gbm_dir.exists() or not any(gbm_dir.glob("*.joblib")):
        print(f"  [skip] no joblibs in {gbm_dir} — run trainer first")
        perm = None
    else:
        perm = perm_importance(df, gbm_dir, n_repeats=args.perm_repeats, sub_n=args.perm_n)

    print("[4/4] plots")
    plot_per_LH(df, out_dir, args.n_layers, args.n_heads)
    plot_y_question_vs_gi(df, out_dir, args.n_layers, args.n_heads)

    payload = {
        "in_dir": str(in_dir),
        "gbm_dir": str(gbm_dir),
        "rows_loaded": int(len(df)),
        "cont_feats": cont_feats,
        "variance_breakdown": vb,
        "univariate_corr": corr,
        "permutation_importance": perm,
    }
    p = out_dir / "summary.json"
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[done] {p}")


if __name__ == "__main__":
    main()
