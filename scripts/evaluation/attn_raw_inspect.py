"""Quick-look + sanity regression on the raw (x, y) parquet outputs.

Reads attn_raw_results/*.parquet, prints schema / row counts / feature ranges,
fits a small per-layer linear regression of y_{sink,sys,docs} on (qprev_len,
log seq_len, docs_total_len, q_rel, sys_len), and prints R^2 per layer to give
a sense of how much variance is explainable from simple features alone.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd


FEATURES = ["qprev_len", "sys_len", "docs_total_len", "n_docs", "q_len", "q_rel"]


def fit_per_layer(df: pd.DataFrame, target: str):
    rows = []
    for L, g in df.groupby("layer"):
        X = g[FEATURES].to_numpy(dtype=np.float64)
        X = np.concatenate([X, np.log1p(g[["seq_len"]].to_numpy(dtype=np.float64))], axis=1)
        X = np.concatenate([X, np.ones((len(g), 1))], axis=1)  # bias
        y = g[target].to_numpy(dtype=np.float64)
        # closed-form OLS
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
        r2 = 1 - ss_res / ss_tot
        rows.append((int(L), float(r2), float(y.mean()), float(y.std())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="attn_raw_results")
    ap.add_argument("--head_subsample", type=int, default=4,
                    help="Use heads % S == 0 only, for speed in the regression")
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        print(f"[empty] no parquet in {in_dir}")
        return

    dfs = []
    for f in files:
        d = pd.read_parquet(f)
        print(f"[load] {f.name}: rows={len(d):,}")
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    print(f"\n[total] {len(df):,} rows, "
          f"benchmarks={sorted(df['benchmark'].unique())}, "
          f"n_layers={int(df['layer'].max())+1}, n_heads={int(df['head'].max())+1}")
    print(df[["qprev_len","sys_len","docs_total_len","n_docs","q_len","seq_len",
              "y_sink","y_sys","y_docs"]].describe().round(3).to_string())

    sub = df[df["head"] % args.head_subsample == 0]
    print(f"\n[regression] subsampled to {len(sub):,} rows "
          f"(every {args.head_subsample}-th head)")
    for target in ["y_sink", "y_sys", "y_docs"]:
        rows = fit_per_layer(sub, target)
        print(f"\n  --- target = {target} (per-layer R^2 of OLS on {FEATURES+['log_seq_len']}) ---")
        print(f"  {'layer':>5} {'R^2':>7} {'mean(y)':>8} {'std(y)':>8}")
        for (L, r2, ym, ys) in rows:
            print(f"  {L:>5} {r2:7.3f} {ym:8.4f} {ys:8.4f}")
        # also benchmark-stratified
        print(f"    benchmark mean(y):")
        for b, g in sub.groupby("benchmark"):
            print(f"      {b:<10} mean={g[target].mean():.4f}  std={g[target].std():.4f}")


if __name__ == "__main__":
    main()
