"""Train + persist HistGradientBoostingRegressor models for the THREE
QUESTION-token attention targets: y_sink / y_sys / y_docs.

This is the clean, block_qa-trained replacement for the original
attn_score_models/{sink,sys,docs}.joblib (which were fit on the eval
benchmarks nq/hotpot/2wiki/musique and therefore leak into downstream eval).

Reads analysis/attention_score_analysis/attn_raw_q_results/*.parquet — question
rows [q_lo,q_hi) collected on block_qa under baseline policy (prompt-only
forward, no generated answer needed; see attn_raw_collect_gen.py --rows question).

Question rows attend over 4 regions: sink / sys-excl / docs / rest(=qprev+self).
The regressor predicts the first three cumulative masses (sink, sys, docs);
rest is the leftover at inference. There is NO separate "question" target here —
the question span is the row's own causal past, i.e. part of "rest".

Feature schema (inference-known at prefill time for a question token at qi):
  raw:     qi, q_len, n_docs, docs_total_len, mean_doc_len, sys_len
  derived: pre_q_len   = sys_len + docs_total_len          (where the question starts)
           cur_pos     = pre_q_len + qi                    (absolute position = qabs)
           qi_to_q_len = qi / max(q_len, 1)
           q_to_docs_ratio = q_len / max(docs_total_len, 1)

Output: analysis/attention_score_analysis/attn_score_q_models/{sink,sys,docs}.joblib
        + summary.json (train / test / heldout R^2).
"""
from __future__ import annotations
import argparse, gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

CONT_FEATS = [
    "qi",
    "q_len",
    "n_docs",
    "docs_total_len",
    "mean_doc_len",
    "sys_len",
    "pre_q_len",
    "cur_pos",
    "qi_to_q_len",
    "q_to_docs_ratio",
]
CAT_FEATS = ["layer", "head"]
TARGETS = ["y_sink", "y_sys", "y_docs"]


def load(in_dir: Path, glob_pattern: str, max_rows: int | None = None,
         seed: int = 0, batch_size: int = 200_000) -> pd.DataFrame:
    """Stream question-row parquets via pyarrow iter_batches with optional
    row-level subsample, then add the derived features."""
    import pyarrow.parquet as pq
    files = sorted(in_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No parquet under {in_dir} matching {glob_pattern!r}")
    total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    keep_frac = (max_rows / total_rows
                 if max_rows and max_rows > 0 and total_rows > max_rows else None)
    print(f"  [load] total_rows={total_rows:,}  keep_frac={keep_frac}")
    rng = np.random.default_rng(seed)
    parts = []
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=batch_size):
            d = batch.to_pandas()
            if keep_frac is not None:
                d = d.sample(frac=keep_frac, random_state=int(rng.integers(0, 2**31 - 1)))
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    del parts; gc.collect()
    for t in TARGETS:
        df[t] = df[t].clip(0.0, 1.0)
    # Derived features (inference-known at prefill time).
    df["pre_q_len"] = (df["sys_len"].astype(np.int32) + df["docs_total_len"].astype(np.int32))
    df["cur_pos"] = df["pre_q_len"] + df["qi"].astype(np.int32)
    df["qi_to_q_len"]     = df["qi"].astype(np.float32) / df["q_len"].clip(lower=1).astype(np.float32)
    df["q_to_docs_ratio"] = df["q_len"].astype(np.float32) / df["docs_total_len"].clip(lower=1).astype(np.float32)
    return df


def split_by_sample(df: pd.DataFrame, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    samples = df["sample_id"].unique().copy()
    rng.shuffle(samples)
    n_test = int(len(samples) * test_frac)
    test_set = set(samples[:n_test])
    test_mask = df["sample_id"].isin(test_set).to_numpy()
    return ~test_mask, test_mask


def r2(y, yhat):
    y = np.asarray(y, np.float64); yhat = np.asarray(yhat, np.float64)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",
                    default="analysis/attention_score_analysis/attn_raw_q_results")
    ap.add_argument("--train_glob", default="block_qa__*.parquet")
    ap.add_argument("--heldout_glob", default="block_qa_test__*.parquet")
    ap.add_argument("--out_dir",
                    default="analysis/attention_score_analysis/attn_score_q_models")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_train_rows", type=int, default=30_000_000)
    ap.add_argument("--max_heldout_rows", type=int, default=90_000_000)
    ap.add_argument("--max_iter", type=int, default=3000)
    ap.add_argument("--learning_rate", type=float, default=0.08)
    ap.add_argument("--max_depth", type=int, default=10)
    ap.add_argument("--min_samples_leaf", type=int, default=200)
    ap.add_argument("--l2_regularization", type=float, default=0.1)
    ap.add_argument("--validation_fraction", type=float, default=0.1)
    ap.add_argument("--n_iter_no_change", type=int, default=30)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    in_dir = Path(args.in_dir)
    df = load(in_dir, args.train_glob, max_rows=args.max_train_rows, seed=args.seed)
    print(f"[load:train] rows={len(df):,}  samples={df['sample_id'].nunique():,}  "
          f"glob={args.train_glob!r}  max_rows={args.max_train_rows}")

    heldout_df = None
    if args.heldout_glob:
        try:
            heldout_df = load(in_dir, args.heldout_glob,
                              max_rows=args.max_heldout_rows, seed=args.seed)
            print(f"[load:heldout] rows={len(heldout_df):,}  "
                  f"samples={heldout_df['sample_id'].nunique():,}  glob={args.heldout_glob!r}")
        except FileNotFoundError:
            print(f"[load:heldout] none matching {args.heldout_glob!r} — skipping heldout R^2.")

    print(f"[feat] cont={CONT_FEATS}  cat={CAT_FEATS}  targets={TARGETS}")
    tr, te = split_by_sample(df, test_frac=args.test_frac, seed=args.seed)
    print(f"[split] train={tr.sum():,}  test={te.sum():,}")

    X = df[CONT_FEATS + CAT_FEATS].to_numpy()
    cat_idx = [len(CONT_FEATS), len(CONT_FEATS) + 1]
    X_ho = heldout_df[CONT_FEATS + CAT_FEATS].to_numpy() if heldout_df is not None else None

    summary = {}
    for target in TARGETS:
        y = df[target].to_numpy(np.float64)
        t0 = time.time()
        m = HistGradientBoostingRegressor(
            max_iter=args.max_iter, max_depth=args.max_depth,
            learning_rate=args.learning_rate, min_samples_leaf=args.min_samples_leaf,
            categorical_features=cat_idx, l2_regularization=args.l2_regularization,
            early_stopping=True, validation_fraction=args.validation_fraction,
            n_iter_no_change=args.n_iter_no_change, random_state=args.seed,
        )
        m.fit(X[tr], y[tr])
        r2_tr = r2(y[tr], m.predict(X[tr]))
        r2_te = r2(y[te], m.predict(X[te]))
        r2_ho = r2(heldout_df[target].to_numpy(np.float64), m.predict(X_ho)) if X_ho is not None else None
        n_used = int(getattr(m, "n_iter_", -1))
        ho_str = f"  heldout_R2={r2_ho:.4f}" if r2_ho is not None else ""
        print(f"  {target:<8} train_R2={r2_tr:.4f}  test_R2={r2_te:.4f}{ho_str}  "
              f"iters={n_used}/{args.max_iter}  [{time.time()-t0:.0f}s]")
        path = out / f"{target.replace('y_', '')}.joblib"
        joblib.dump({
            "model": m, "cont_feats": CONT_FEATS, "cat_feats": CAT_FEATS,
            "target": target, "r2_train": float(r2_tr), "r2_test": float(r2_te),
            "r2_heldout": float(r2_ho) if r2_ho is not None else None, "n_iter": n_used,
        }, path, compress=3)
        summary[target] = {"r2_train": float(r2_tr), "r2_test": float(r2_te),
                           "r2_heldout": float(r2_ho) if r2_ho is not None else None,
                           "n_iter": n_used, "path": str(path)}

    with open(out / "summary.json", "w") as f:
        json.dump({"cont_feats": CONT_FEATS, "cat_feats": CAT_FEATS,
                   "test_frac": args.test_frac, "train_glob": args.train_glob,
                   "heldout_glob": args.heldout_glob,
                   "max_train_rows": args.max_train_rows,
                   "train_rows": int(len(df)),
                   "heldout_rows": int(len(heldout_df)) if heldout_df is not None else 0,
                   "results": summary}, f, indent=2)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
