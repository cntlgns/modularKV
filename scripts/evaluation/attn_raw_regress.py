"""Regress baseline-policy attention masses against (sample,layer,head,qi)
features. Targets: y_sink, y_sys, y_docs.

Compares 6 modeling approaches on a sample-stratified 80/20 train/test split:
  M1: pooled OLS, continuous features only          (head/layer pooled away)
  M2: pooled OLS, continuous + one-hot(layer,head,benchmark)
  M3: pooled HistGradientBoosting, continuous + cat(layer,head,benchmark)
  M4: pooled HistGradientBoosting on logit(y) target (helps for [0,1] targets)
  M5: per-(layer,head) OLS, continuous features
  M6: per-(layer,head) HistGradientBoosting, continuous features

Writes per-(layer,head) R² heatmaps and a summary.json.
"""
from __future__ import annotations
import argparse, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

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


def r2(y, yhat):
    y = np.asarray(y, dtype=np.float64); yhat = np.asarray(yhat, dtype=np.float64)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def split_by_sample(df, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    samples = df["sample_id"].unique().copy()
    rng.shuffle(samples)
    n_test = int(len(samples) * test_frac)
    test_set = set(samples[:n_test])
    test_mask = df["sample_id"].isin(test_set).to_numpy()
    return ~test_mask, test_mask


# ---------- pooled OLS continuous ---------- #
def m1_ols_cont(df, tr, te, target):
    X = df[CONT_FEATS].to_numpy(np.float64)
    X = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    y = df[target].to_numpy(np.float64)
    beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
    return r2(y[tr], X[tr] @ beta), r2(y[te], X[te] @ beta)


# ---------- pooled OLS + one-hot ---------- #
# NOTE: benchmark is intentionally EXCLUDED from feature set (we want a
# correction that's benchmark-agnostic — sample-level token length features
# carry all the relevant "context size" info). benchmark stays a column in
# the parquet only for diagnostic stratification.
def m2_ols_dummies(df, tr, te, target):
    parts = [df[CONT_FEATS].astype(np.float32).to_numpy()]
    for col in ["layer", "head"]:
        d = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=np.float32).to_numpy()
        parts.append(d)
    X = np.concatenate(parts + [np.ones((len(df), 1), dtype=np.float32)], axis=1).astype(np.float64)
    y = df[target].to_numpy(np.float64)
    beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
    return r2(y[tr], X[tr] @ beta), r2(y[te], X[te] @ beta), X.shape[1]


# ---------- pooled HistGBM ---------- #
def m3_gbm(df, tr, te, target):
    X = df[CONT_FEATS + ["layer", "head"]].copy()
    X_arr = X.to_numpy()
    y = df[target].to_numpy(np.float64)
    cat_idx = [len(CONT_FEATS), len(CONT_FEATS) + 1]
    m = HistGradientBoostingRegressor(
        max_iter=300, max_depth=8, learning_rate=0.08, min_samples_leaf=200,
        categorical_features=cat_idx, l2_regularization=0.1, random_state=0,
    )
    m.fit(X_arr[tr], y[tr])
    return r2(y[tr], m.predict(X_arr[tr])), r2(y[te], m.predict(X_arr[te]))


# ---------- pooled HistGBM on logit(y) ---------- #
def m4_gbm_logit(df, tr, te, target):
    eps = 1e-3
    y_raw = df[target].to_numpy(np.float64).clip(eps, 1 - eps)
    y_logit = np.log(y_raw / (1 - y_raw))
    X = df[CONT_FEATS + ["layer", "head"]].copy()
    X_arr = X.to_numpy()
    cat_idx = [len(CONT_FEATS), len(CONT_FEATS) + 1]
    m = HistGradientBoostingRegressor(
        max_iter=300, max_depth=8, learning_rate=0.08, min_samples_leaf=200,
        categorical_features=cat_idx, l2_regularization=0.1, random_state=0,
    )
    m.fit(X_arr[tr], y_logit[tr])
    yhat_tr = 1.0 / (1.0 + np.exp(-m.predict(X_arr[tr])))
    yhat_te = 1.0 / (1.0 + np.exp(-m.predict(X_arr[te])))
    # Report R^2 in original (probability) space
    y = df[target].to_numpy(np.float64)
    return r2(y[tr], yhat_tr), r2(y[te], yhat_te)


# ---------- per-(layer, head) ---------- #
def m5_m6_per_layer_head(df, tr, te, target, kind, n_layers, n_heads):
    """For each (L, h): fit on train slice, evaluate on test slice. kind='ols' or 'gbm'."""
    layers = df["layer"].to_numpy()
    heads = df["head"].to_numpy()
    X_full = df[CONT_FEATS].to_numpy(np.float64)
    y_full = df[target].to_numpy(np.float64)
    out = np.full((n_layers, n_heads), np.nan, dtype=np.float64)

    # Pre-build per-(L,h) index arrays for speed
    for L in range(n_layers):
        for H in range(n_heads):
            mask = (layers == L) & (heads == H)
            if mask.sum() < 50:
                continue
            tr_lh = mask & tr
            te_lh = mask & te
            if te_lh.sum() < 10 or tr_lh.sum() < 20:
                continue
            Xtr = X_full[tr_lh]; Xte = X_full[te_lh]
            ytr = y_full[tr_lh]; yte = y_full[te_lh]
            if kind == "ols":
                Xtr_b = np.concatenate([Xtr, np.ones((len(Xtr), 1))], axis=1)
                Xte_b = np.concatenate([Xte, np.ones((len(Xte), 1))], axis=1)
                beta, *_ = np.linalg.lstsq(Xtr_b, ytr, rcond=None)
                yhat = Xte_b @ beta
            else:
                m = HistGradientBoostingRegressor(
                    max_iter=80, max_depth=4, learning_rate=0.1,
                    min_samples_leaf=20, l2_regularization=1.0, random_state=0,
                )
                m.fit(Xtr, ytr)
                yhat = m.predict(Xte)
            out[L, H] = r2(yte, yhat)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="attn_raw_results")
    ap.add_argument("--out_dir", default="attn_regress_results")
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--skip_perLH_gbm", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    df = load(Path(args.in_dir))
    print(f"[load] rows={len(df):,}  feats={CONT_FEATS}")
    n_layers = int(df["layer"].max()) + 1
    n_heads  = int(df["head"].max()) + 1
    tr, te = split_by_sample(df, test_frac=0.2, seed=0)
    print(f"[split] train={tr.sum():,} test={te.sum():,} ({n_layers} layers x {n_heads} heads)")

    summary = {}
    for target in args.targets:
        print(f"\n========== target={target} ==========")
        t0 = time.time(); r2_tr, r2_te = m1_ols_cont(df, tr, te, target)
        print(f"  M1 OLS continuous              train_R2={r2_tr:.4f}  test_R2={r2_te:.4f}   [{time.time()-t0:.0f}s]")
        t0 = time.time(); r2_tr2, r2_te2, k = m2_ols_dummies(df, tr, te, target)
        print(f"  M2 OLS + one-hot(L,h)          train_R2={r2_tr2:.4f}  test_R2={r2_te2:.4f}  d={k} [{time.time()-t0:.0f}s]")
        t0 = time.time(); r2_tr3, r2_te3 = m3_gbm(df, tr, te, target)
        print(f"  M3 HistGBM raw target          train_R2={r2_tr3:.4f}  test_R2={r2_te3:.4f}   [{time.time()-t0:.0f}s]")
        t0 = time.time(); r2_tr4, r2_te4 = m4_gbm_logit(df, tr, te, target)
        print(f"  M4 HistGBM logit(y)            train_R2={r2_tr4:.4f}  test_R2={r2_te4:.4f}   [{time.time()-t0:.0f}s]")
        t0 = time.time(); r2_ols = m5_m6_per_layer_head(df, tr, te, target, "ols", n_layers, n_heads)
        print(f"  M5 per-(L,h) OLS               test_R2 mean={np.nanmean(r2_ols):.4f}  med={np.nanmedian(r2_ols):.4f}  p90={np.nanpercentile(r2_ols, 90):.4f}  max={np.nanmax(r2_ols):.4f}  [{time.time()-t0:.0f}s]")
        np.save(out / f"r2_perLH_ols_{target}.npy", r2_ols)
        r2_gbm = None
        if not args.skip_perLH_gbm:
            t0 = time.time(); r2_gbm = m5_m6_per_layer_head(df, tr, te, target, "gbm", n_layers, n_heads)
            print(f"  M6 per-(L,h) HistGBM           test_R2 mean={np.nanmean(r2_gbm):.4f}  med={np.nanmedian(r2_gbm):.4f}  p90={np.nanpercentile(r2_gbm, 90):.4f}  max={np.nanmax(r2_gbm):.4f}  [{time.time()-t0:.0f}s]")
            np.save(out / f"r2_perLH_gbm_{target}.npy", r2_gbm)
        summary[target] = {
            "M1_ols_cont":       float(r2_te),
            "M2_ols_dummies":    float(r2_te2),
            "M3_gbm":            float(r2_te3),
            "M4_gbm_logit":      float(r2_te4),
            "M5_perLH_ols_mean": float(np.nanmean(r2_ols)),
            "M5_perLH_ols_p90":  float(np.nanpercentile(r2_ols, 90)),
            "M6_perLH_gbm_mean": float(np.nanmean(r2_gbm)) if r2_gbm is not None else None,
            "M6_perLH_gbm_p90":  float(np.nanpercentile(r2_gbm, 90)) if r2_gbm is not None else None,
        }

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Heatmaps of per-(L, h) test R^2
    figs = out / "figs"; figs.mkdir(exist_ok=True)
    for target in args.targets:
        for kind in (["ols"] if args.skip_perLH_gbm else ["ols", "gbm"]):
            arr = np.load(out / f"r2_perLH_{kind}_{target}.npy")
            arr_show = arr.copy(); arr_show[np.isnan(arr_show)] = 0.0
            fig, ax = plt.subplots(figsize=(7, 5.5))
            vmax = max(0.3, float(np.nanmax(arr)))
            im = ax.imshow(arr_show, aspect="auto", vmin=0, vmax=vmax, cmap="viridis")
            ax.set_xlabel("head"); ax.set_ylabel("layer")
            ax.set_title(f"{target}: per-(layer, head) test R²  [{kind}]\n"
                         f"mean={np.nanmean(arr):.3f}  median={np.nanmedian(arr):.3f}  max={np.nanmax(arr):.3f}")
            plt.colorbar(im, ax=ax, label="R²")
            fig.tight_layout()
            fig.savefig(figs / f"r2_perLH_{kind}_{target}.png", dpi=120)
            plt.close(fig)
    print(f"\n[done] summary -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
