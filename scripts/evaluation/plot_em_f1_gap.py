"""Plot the substring-accuracy vs EM/F1 gap for the KVLink finetune.

Message: after KVLink finetuning the model recovers *substring accuracy*
(the gold span appears somewhere in the output) but NOT EM/F1, because the
training target is the verbose GPT-generated full sentence, not the short answer.

Reads result/ft_vs_pt/comparison_long.csv and writes analysis/em_f1_gap.png.
"""
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = "result/ft_vs_pt/comparison_long.csv"

# benchmark key -> nice label
BENCHES = [("NQ", "NQ"), ("hqa", "HotpotQA"), ("wiki", "2WikiMQA"), ("musique", "MuSiQue")]

# We focus on the cleanest single-doc setting: goldonly, short prompt, recover_pos_enc.
SEL = dict(state="goldonly", prompt_preset="short", kv_policy="recover_pos_enc")
FT_MODEL = "Llama-3.2-1B-Instruct-ft-step6000"
PT_MODEL = "Llama-3.2-1B-Instruct"  # pretrained, recover_pos_enc (broken policy)


def load():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def pick(rows, bench, model, source):
    for r in rows:
        if (
            r["benchmark"] == bench
            and r["model"] == model
            and r["source"] == source
            and r["state"] == SEL["state"]
            and r["prompt_preset"] == SEL["prompt_preset"]
            and r["kv_policy"] == SEL["kv_policy"]
        ):
            return float(r["accuracy"]), float(r["em"]), float(r["f1"])
    return None


rows = load()
labels = [lbl for _, lbl in BENCHES]
acc, em, f1 = [], [], []
for key, _ in BENCHES:
    a, e, f = pick(rows, key, FT_MODEL, "ft_vs_pt_n1000")
    acc.append(a); em.append(e); f1.append(f)

x = np.arange(len(labels))
w = 0.26

fig, ax = plt.subplots(figsize=(9, 5.2))
b1 = ax.bar(x - w, acc, w, label="Substring accuracy", color="#2e7d32")
b2 = ax.bar(x, f1, w, label="F1", color="#f9a825")
b3 = ax.bar(x + w, em, w, label="Exact Match (EM)", color="#c62828")

for bars in (b1, b2, b3):
    for r in bars:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h + 0.012, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)

ax.set_ylabel("Score")
ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title(
    "KVLink finetune (step-6000, short prompt, gold doc, recover_pos_enc)\n"
    "Substring accuracy recovers — EM / F1 do not",
    fontsize=12,
)
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
ax.text(
    0.0, -0.16,
    "Cause: training target = GPT-generated full-sentence answer "
    "(e.g. \"Yes, both ... are located in Iran.\"), not the short gold span (\"yes\").\n"
    "The gold span is contained in the verbose output (high substring acc) but the output "
    "is never the exact short answer (EM~0, low F1).",
    transform=ax.transAxes, fontsize=8.5, color="#444",
)

fig.tight_layout()
out = "analysis/em_f1_gap.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
