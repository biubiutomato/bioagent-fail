#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

summary = json.loads((ROOT / "results" / "summary_by_policy.json").read_text())
by_fault = json.loads((ROOT / "results" / "summary_by_fault.json").read_text())

order = ["naive", "retry_on_error", "selfcheck", "verify_or_abstain"]
label = {
    "naive": "Naive",
    "retry_on_error": "Retry-on-error",
    "selfcheck": "Self-check",
    "verify_or_abstain": "Verify-or-abstain",
}
smap = {s["policy"]: s for s in summary}

plt.rcParams.update({
    # IEEE PDF eXpress rejects Type 3 fonts: embed TrueType (Type 42) instead.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(5.2, 2.4))
x = np.arange(len(order))
w = 0.22
metrics = [("exact_match", "Exact match"), ("safe", "Safe terminal"), ("unsafe_accept", "Unsafe accept")]
cols = ["#1f4e79", "#2e8b57", "#c0392b"]
for i, (k, lab) in enumerate(metrics):
    vals = [smap[p][k] * 100 for p in order]
    ax.bar(x + (i - 1) * w, vals, w, label=lab, color=cols[i])
ax.set_xticks(x)
ax.set_xticklabels([label[p] for p in order], rotation=12)
ax.set_ylabel("Rate (%)")
ax.set_ylim(0, 110)
ax.legend(frameon=False, loc="upper left")
ax.set_title("Terminal decision quality on BioAgent-Fail (64 tasks)")
fig.savefig(FIG / "policy_overview.pdf", metadata={"Creator": None, "Producer": None})
fig.savefig(FIG / "policy_overview.png")
plt.close()

faults = ["clean", "missing_file", "empty_output", "nonzero_exit", "timeout", "corrupted_output", "semantic_wrong", "version_mismatch"]
fault_lab = {
    "clean": "Clean",
    "missing_file": "Missing",
    "empty_output": "Empty",
    "nonzero_exit": "Exit≠0",
    "timeout": "Timeout",
    "corrupted_output": "Corrupt",
    "semantic_wrong": "Semantic",
    "version_mismatch": "Build",
}
mat = np.zeros((len(order), len(faults)))
for i, p in enumerate(order):
    for j, f in enumerate(faults):
        rec = next(r for r in by_fault if r["policy"] == p and r["fault"] == f)
        mat[i, j] = rec["unsafe_accept"] * 100

fig, ax = plt.subplots(figsize=(5.4, 2.3))
im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(faults)))
ax.set_xticklabels([fault_lab[f] for f in faults], rotation=30, ha="right")
ax.set_yticks(range(len(order)))
ax.set_yticklabels([label[p] for p in order])
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", color="white" if mat[i, j] > 55 else "black", fontsize=6.5)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Unsafe accept (%)")
ax.set_title("Unsafe continuation by failure class")
fig.savefig(FIG / "unsafe_heatmap.pdf", metadata={"Creator": None, "Producer": None})
fig.savefig(FIG / "unsafe_heatmap.png")
plt.close()
print("wrote", FIG)
