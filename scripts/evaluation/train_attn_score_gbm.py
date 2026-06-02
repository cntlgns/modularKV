"""Train + persist HistGradientBoostingRegressor models for y_sink / y_sys / y_docs.

Reads attn_raw_results/*.parquet (baseline-policy raw data) and fits 3 models
on the SAME train split as attn_raw_regress.py M3. Saves to
attn_score_models/{sink,sys,docs}.joblib so the eval script can load them.

Feature schema mirrors attn_raw_regress.CONT_FEATS + categorical (layer, head).
Benchmark is NOT used (per user request).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

CONT_FEATS = [
    "qprev_len", "docs_total_len", "n_docs", "mean_doc_len",
    "min_doc_len", "max_doc_len", "q_len", "seq_len", "q_rel", "qabs",
    "log_seq_len", "log1p_docs_total_len", "log1p_qabs", "docs_to_seq_ratio",
]
TARGETS = ["y_sink", "y_sys", "y_docs"]


def load(in_dir: Path) -> pd.DataFrame:
    files = sorted(in_dir.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["log_seq_len"]          = np.log(df["seq_len"].astype("float32"))
    df["log1p_docs_total_len"] = np.log1p(df["docs_total_len"].astype("float32"))
    df["log1p_qabs"]           = np.log1p(df["qabs"].astype("float32"))
    df["docs_to_seq_ratio"]    = (df["docs_total_len"].astype("float32")
                                  / df["seq_len"].astype("float32"))
    for t in TARGETS:
        df[t] = df[t].clip(0.0, 1.0)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="attn_raw_results")
    ap.add_argument("--out_dir", default="attn_score_models")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    df = load(Path(args.in_dir))
    print(f"[load] rows={len(df):,}")

    # sample-stratified train/test split (matching attn_raw_regress.py)
    rng = np.random.default_rng(args.seed)
    samples = df["sample_id"].unique().copy()
    rng.shuffle(samples)
    n_test = int(len(samples) * args.test_frac)
    test_set = set(samples[:n_test])
    test_mask = df["sample_id"].isin(test_set).to_numpy()
    tr = ~test_mask
    print(f"[split] train={tr.sum():,} test={test_mask.sum():,}")

    X = df[CONT_FEATS + ["layer", "head"]].to_numpy()
    cat_idx = [len(CONT_FEATS), len(CONT_FEATS) + 1]

    summary = {}
    for target in TARGETS:
        y = df[target].to_numpy(np.float64)
        t0 = time.time()
        m = HistGradientBoostingRegressor(
            max_iter=300, max_depth=8, learning_rate=0.08,
            min_samples_leaf=200, categorical_features=cat_idx,
            l2_regularization=0.1, random_state=0,
        )
        m.fit(X[tr], y[tr])
        yhat_tr = m.predict(X[tr])
        yhat_te = m.predict(X[test_mask])
        r2_tr = 1 - np.sum((y[tr] - yhat_tr)**2) / (np.sum((y[tr] - y[tr].mean())**2) + 1e-12)
        r2_te = 1 - np.sum((y[test_mask] - yhat_te)**2) / (np.sum((y[test_mask] - y[test_mask].mean())**2) + 1e-12)
        print(f"  {target:<8} train_R2={r2_tr:.4f}  test_R2={r2_te:.4f}  [{time.time()-t0:.0f}s]")
        path = out / f"{target.replace('y_','')}.joblib"
        joblib.dump({
            "model": m,
            "cont_feats": CONT_FEATS,
            "cat_feats": ["layer", "head"],
            "target": target,
            "r2_train": float(r2_tr),
            "r2_test": float(r2_te),
        }, path, compress=3)
        summary[target] = {"r2_train": float(r2_tr), "r2_test": float(r2_te), "path": str(path)}

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
