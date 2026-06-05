"""Print the oracle_ab_vrec K-vs-V comparison table from result summaries.

Reads result/oracle_ab_vrec/*.summary.json and lays out, per (benchmark, mode),
the three policies side by side:

    baseline  (ceiling, full cache)
    oracle    (oracle_ab_vrec = baseline attention scores x recover values)
    broken    (recover_pos_enc, block-diagonal cache)

Recovery fraction r = (oracle - broken) / (baseline - broken) on substring-acc:
r~1 => the damage was all in K (routing), perfect attention recovers it;
r~0 => the damage is in V (content), attention correction cannot help.
"""
import glob
import json
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "result", "oracle_ab_vrec")
BENCHES = ["NQ", "hqa", "wiki", "musique"]
MODES = [("all", ["at0", "all"]), ("goldonly", ["goldonly"])]
# short label per policy. ab_vrec = (A_base, V_rec) "perfect attn"; arec_vbase
# = (A_rec, V_base) "perfect value".
POLICIES = [
    ("baseline", "baseline"),
    ("oracle_ab_vrec", "A_base*Vrec"),
    ("oracle_arec_vbase", "Arec*Vbase"),
    ("recover_pos_enc", "broken"),
]


def load():
    out = {}
    for f in glob.glob(os.path.join(OUT_DIR, "*.summary.json")):
        d = json.load(open(f))
        out[(d["benchmark"], d["state"], d["kv_policy"])] = d
    return out


def main():
    rows = load()
    print(f"\nloaded {len(rows)} summaries from {os.path.normpath(OUT_DIR)}")
    print("recovK% = (A_base*Vrec - broken)/(baseline - broken)  [does fixing K suffice?]")
    print("recovV% = (Arec*Vbase - broken)/(baseline - broken)  [does fixing V suffice?]\n")
    hdr = (f"{'bench':8s} {'mode':5s} {'metric':4s} | {'baseline':>9s} "
           f"{'A_base*Vrec':>11s} {'Arec*Vbase':>11s} {'broken':>9s} | {'recovK%':>7s} {'recovV%':>7s}")
    print(hdr); print("-" * len(hdr))
    for bench in BENCHES:
        for mode_name, states in MODES:
            got = {}
            n = None
            for state in states:
                for pol, short in POLICIES:
                    d = rows.get((bench, state, pol))
                    if d:
                        got[short] = d
                        n = d["num_examples"]
            if not got:
                print(f"{bench:8s} {mode_name:5s}  (no results yet)")
                continue
            for metric in ("accuracy", "em", "f1"):
                b = got.get("baseline", {}).get(metric)
                av = got.get("A_base*Vrec", {}).get(metric)
                rv = got.get("Arec*Vbase", {}).get(metric)
                k = got.get("broken", {}).get(metric)
                def fmt(x, w=9): return f"{x:{w}.3f}" if isinstance(x, (int, float)) else f"{'-':>{w}s}"
                rk = rvp = ""
                if metric == "accuracy" and None not in (b, k) and abs(b - k) > 1e-9:
                    if av is not None:
                        rk = f"{100*(av-k)/(b-k):7.1f}"
                    if rv is not None:
                        rvp = f"{100*(rv-k)/(b-k):7.1f}"
                tag = f"(n={n})" if metric == "accuracy" else ""
                print(f"{bench:8s} {mode_name:5s} {metric:4s} | {fmt(b)} {fmt(av,11)} "
                      f"{fmt(rv,11)} {fmt(k)} | {rk:>7s} {rvp:>7s} {tag}")
            print()


if __name__ == "__main__":
    main()
