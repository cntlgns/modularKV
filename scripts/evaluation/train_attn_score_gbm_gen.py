"""Train + persist HistGradientBoostingRegressor models for the FOUR
generation-token attention targets: y_sink / y_sys / y_docs / y_question.

Reads analysis/attention_score_analysis/attn_raw_gen_results/*.parquet (baseline-policy, gen-row data collected
on block_qa train via teacher-forcing the GPT-4o-mini `generated` field) and
fits 4 GBMs with EARLY STOPPING (validation_fraction=0.1, n_iter_no_change=20).

Compared to the question-token GBMs (attn_score_models/{sink,sys,docs}.joblib),
this version:
  - Adds a 4th target (y_question) — question span is now a separate region
    in the rescaler, not bundled into "rest".
  - Trims the feature set from 14 cont feats to 7, dropping redundant log /
    ratio derivatives (HistGBM is invariant to monotone transforms anyway)
    and the min/max doc length variants (mean_doc + n_docs captures most).
  - Enables sklearn's internal validation_fraction-based early stopping so
    `max_iter` is no longer a hard hyperparameter the user has to tune by eye.

Output: analysis/attention_score_analysis/attn_score_gen_models/{sink,sys,docs,question}.joblib + summary.json.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor


# Feature set v2 — extends v1 (7 raw cont) with 4 derived cross-features that
# combine row-level (gi) and sample-level (q_len, docs_total_len, ...) info.
# Derived feats are computed in load() so the saved joblib carries the schema.
#
# All features are quantities **known at decode time** for a given gi step:
#   * Sample-level: sys_len, n_docs, docs_total_len, mean_doc_len, q_len (after prefill)
#   * Row-level: gi (current decode step)
#   * prompt_len = sys_len + docs_total_len + q_len   (post-prefill, constant per sample)
#   * cur_pos    = prompt_len + gi                    (current absolute position)
#   * q_to_prompt_ratio = q_len / prompt_len          (how big is the question slice)
#   * q_to_docs_ratio   = q_len / docs_total_len
#   * gi_to_q_len       = gi / max(q_len, 1)          (gen depth normalized by q size)
#
# NOTE: we dropped raw `seq_len` (final sequence length) — at training it includes
# the full teacher-forced gen tail, but at inference for step gi the cache only
# contains prompt_len + gi tokens. cur_pos replaces it with the inference-correct
# value.
CONT_FEATS = [
    "gi",
    "q_len",
    "n_docs",
    "docs_total_len",
    "mean_doc_len",
    "sys_len",
    "prompt_len",
    "cur_pos",
    "q_to_prompt_ratio",
    "q_to_docs_ratio",
    "gi_to_q_len",
]
CAT_FEATS = ["layer", "head"]
TARGETS = ["y_sink", "y_sys", "y_docs", "y_question"]


def load(in_dir: Path, glob_pattern: str = "*.parquet",
         max_rows: int | None = None, seed: int = 0,
         batch_size: int = 200_000) -> pd.DataFrame:
    """Stream parquets matching ``glob_pattern`` via pyarrow ``iter_batches`` and
    return a single concatenated, target-clipped DataFrame.

    If ``max_rows`` is set and the parquets contain more total rows than that, each
    batch is uniformly sub-sampled at the **row** level so the final DataFrame
    contains roughly ``max_rows`` rows. We subsample at the row level rather than at
    the sample-id level because (a) the ~400M-row sample-id scan blows up Python
    string memory (~80 bytes per str object x 400M = ~30GB just for the column)
    and (b) HistGBM doesn't care about per-sample row counts — it treats each row
    as an independent training example.

    Memory ceiling per batch is ~200K rows x 80 bytes ~= 16 MB, so this fits easily
    in tight SLURM allocations.
    """
    import gc
    import pyarrow.parquet as pq

    files = sorted(in_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No parquet under {in_dir} matching {glob_pattern!r}")

    total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    keep_frac = None
    if max_rows is not None and max_rows > 0 and total_rows > max_rows:
        keep_frac = max_rows / total_rows
    print(f"  [load] total_rows={total_rows:,}  keep_frac={keep_frac}")

    rng = np.random.default_rng(seed)
    parts = []
    for f in files:
        pq_file = pq.ParquetFile(f)
        for batch in pq_file.iter_batches(batch_size=batch_size):
            df_batch = batch.to_pandas()
            if keep_frac is not None:
                # Per-batch seed so each batch is reproducible without retaining
                # a single rng across the loop body (which would otherwise spread
                # 'sample' state and fight pandas's own random_state argument).
                bseed = int(rng.integers(0, 2**31 - 1))
                df_batch = df_batch.sample(frac=keep_frac, random_state=bseed)
            parts.append(df_batch)

    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    for t in TARGETS:
        df[t] = df[t].clip(0.0, 1.0)

    # Derived features (cf. CONT_FEATS docstring). All inference-known quantities.
    df["prompt_len"] = (df["sys_len"].astype(np.int32) +
                        df["docs_total_len"].astype(np.int32) +
                        df["q_len"].astype(np.int32))
    df["cur_pos"] = (df["prompt_len"] + df["gi"].astype(np.int32))
    df["q_to_prompt_ratio"] = df["q_len"].astype(np.float32) / df["prompt_len"].clip(lower=1).astype(np.float32)
    df["q_to_docs_ratio"]   = df["q_len"].astype(np.float32) / df["docs_total_len"].clip(lower=1).astype(np.float32)
    df["gi_to_q_len"]       = df["gi"].astype(np.float32) / df["q_len"].clip(lower=1).astype(np.float32)
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
    y = np.asarray(y, dtype=np.float64); yhat = np.asarray(yhat, dtype=np.float64)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",
                    default="analysis/attention_score_analysis/attn_raw_gen_results")
    ap.add_argument("--train_glob", default="block_qa__*.parquet",
                    help="Glob for training-split parquets under --in_dir. "
                         "Default excludes 'block_qa_test__*.parquet' so the heldout split "
                         "isn't accidentally folded into training.")
    ap.add_argument("--heldout_glob", default="block_qa_test__*.parquet",
                    help="Glob for the heldout block_qa test parquets under --in_dir. "
                         "If any match, the trainer reports a third R^2 column "
                         "('heldout_R2') in addition to the train-split-stratified test "
                         "R^2. Set to '' to skip heldout evaluation.")
    ap.add_argument("--out_dir",
                    default="analysis/attention_score_analysis/attn_score_gen_models")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed",      type=int,   default=0)
    ap.add_argument("--max_train_rows", type=int, default=30_000_000,
                    help="Cap on rows kept from the train-split parquets. Each row is "
                         "~80 bytes in pandas, so 30M rows ~= ~2.5GB DataFrame and "
                         "trains 4 GBMs in <1h on 10 CPUs. Bump to 100M for more data if "
                         "R^2 is poor (still ~8GB / fits in 60GB allocation but each "
                         "target then takes ~15-20 min, so 4 targets ~= 1-1.5h). Set to 0 "
                         "to disable subsampling and load all ~800M rows (needs ~80GB RAM).")
    ap.add_argument("--max_heldout_rows", type=int, default=90_000_000,
                    help="Cap on rows kept from the heldout parquets. The test split's "
                         "~90M rows fit fine, so this is effectively a no-op at default.")
    # GBM hyperparameters (with early stopping ⇒ max_iter is a ceiling, not a knob).
    ap.add_argument("--max_iter",         type=int,   default=3000)
    ap.add_argument("--learning_rate",    type=float, default=0.08)
    ap.add_argument("--max_depth",        type=int,   default=10)
    ap.add_argument("--min_samples_leaf", type=int,   default=200)
    ap.add_argument("--l2_regularization", type=float, default=0.1)
    ap.add_argument("--validation_fraction", type=float, default=0.1)
    ap.add_argument("--n_iter_no_change",    type=int,   default=30)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(exist_ok=True, parents=True)
    in_dir = Path(args.in_dir)
    df = load(in_dir, args.train_glob, max_rows=args.max_train_rows, seed=args.seed)
    print(f"[load:train] rows={len(df):,}  unique samples={df['sample_id'].nunique():,}  "
          f"glob={args.train_glob!r}  max_rows={args.max_train_rows}")

    # Optional independent heldout pool (block_qa test split parquets).
    heldout_df = None
    if args.heldout_glob:
        try:
            heldout_df = load(in_dir, args.heldout_glob,
                              max_rows=args.max_heldout_rows, seed=args.seed)
            print(f"[load:heldout] rows={len(heldout_df):,}  "
                  f"unique samples={heldout_df['sample_id'].nunique():,}  "
                  f"glob={args.heldout_glob!r}  max_rows={args.max_heldout_rows}")
        except FileNotFoundError:
            print(f"[load:heldout] no parquets matching {args.heldout_glob!r} — "
                  f"skipping heldout R^2 (train-split-stratified test R^2 only).")

    print(f"[feat] cont={CONT_FEATS}  cat={CAT_FEATS}  targets={TARGETS}")

    tr, te = split_by_sample(df, test_frac=args.test_frac, seed=args.seed)
    print(f"[split] train={tr.sum():,}  test={te.sum():,}")

    X = df[CONT_FEATS + CAT_FEATS].to_numpy()
    cat_idx = [len(CONT_FEATS), len(CONT_FEATS) + 1]
    X_heldout = (heldout_df[CONT_FEATS + CAT_FEATS].to_numpy()
                 if heldout_df is not None else None)

    summary = {}
    for target in TARGETS:
        y = df[target].to_numpy(np.float64)
        t0 = time.time()
        m = HistGradientBoostingRegressor(
            max_iter=args.max_iter,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            min_samples_leaf=args.min_samples_leaf,
            categorical_features=cat_idx,
            l2_regularization=args.l2_regularization,
            early_stopping=True,
            validation_fraction=args.validation_fraction,
            n_iter_no_change=args.n_iter_no_change,
            random_state=args.seed,
        )
        m.fit(X[tr], y[tr])
        yhat_tr = m.predict(X[tr])
        yhat_te = m.predict(X[te])
        r2_tr = r2(y[tr], yhat_tr)
        r2_te = r2(y[te], yhat_te)
        dt = time.time() - t0
        n_used = int(getattr(m, "n_iter_", -1))

        r2_ho = None
        if X_heldout is not None:
            y_ho = heldout_df[target].to_numpy(np.float64)
            r2_ho = r2(y_ho, m.predict(X_heldout))

        ho_str = f"  heldout_R2={r2_ho:.4f}" if r2_ho is not None else ""
        print(f"  {target:<11} train_R2={r2_tr:.4f}  test_R2={r2_te:.4f}{ho_str}  "
              f"iters={n_used}/{args.max_iter}  [{dt:.0f}s]")
        path = out / f"{target.replace('y_', '')}.joblib"
        joblib.dump({
            "model": m,
            "cont_feats": CONT_FEATS,
            "cat_feats":  CAT_FEATS,
            "target": target,
            "r2_train":   float(r2_tr),
            "r2_test":    float(r2_te),
            "r2_heldout": float(r2_ho) if r2_ho is not None else None,
            "n_iter":     n_used,
        }, path, compress=3)
        summary[target] = {
            "r2_train": float(r2_tr), "r2_test": float(r2_te),
            "r2_heldout": float(r2_ho) if r2_ho is not None else None,
            "n_iter": n_used, "path": str(path),
        }

    with open(out / "summary.json", "w") as f:
        json.dump({
            "cont_feats":          CONT_FEATS,
            "cat_feats":           CAT_FEATS,
            "test_frac":           args.test_frac,
            "train_glob":          args.train_glob,
            "heldout_glob":        args.heldout_glob,
            "max_train_rows":   args.max_train_rows,
            "max_heldout_rows": args.max_heldout_rows,
            "train_rows":          int(len(df)),
            "train_unique_samples": int(df["sample_id"].nunique()),
            "heldout_rows":        int(len(heldout_df)) if heldout_df is not None else 0,
            "heldout_unique_samples": int(heldout_df["sample_id"].nunique()) if heldout_df is not None else 0,
            "results":             summary,
        }, f, indent=2)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
