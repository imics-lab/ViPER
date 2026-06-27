"""
make_figure2_training_curves.py
─────────────────────────────────────────────────────────────────
Real Figure 2 replacement: BloodMNIST training curves.

Reads per-epoch history from viper_v5_results_bloodmnist/runs/<method>_seed42.json
(history.train_loss and history.val[].acc) and produces a single-panel
validation-accuracy plot identical in style to the current Figure 2,
but with REAL numbers and only the methods we actually trained.

Methods plotted (only if JSON exists):
  - No PE       (none_seed42.json)
  - APE         (learned_seed42.json)
  - CPE         (cpe_seed42.json)
  - ViPER       (viper_v5_seed42.json)

The old Figure 2 included a fictional "WPE" line; that is omitted here.

Usage:
  cd ~/ViPER/notebooks
  python3 make_figure2_training_curves.py

Output: f2_training_curves.pdf  (also saves a .png at 150 DPI)
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

# ─── Publication styling ──────────────────────────────────────────
mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":    9,
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "grid.alpha":        0.30,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# ─── Configuration ───────────────────────────────────────────────
RUNS_DIR = "viper_v5_results_bloodmnist/runs"
SEED     = 42

# Order matters for legend / draw order (later = on top)
METHODS = [
    ("none_seed42",      "No PE",        "#9CA3AF", 1.4),
    ("learned_seed42",   "APE",          "#A78BFA", 1.4),
    ("cpe_seed42",       "CPE",          "#60A5FA", 1.4),
    ("viper_v5_seed42",  "ViPER (ours)", "#DC2626", 2.2),
]

# ─── Load runs ───────────────────────────────────────────────────
loaded = []
missing = []
for stem, label, color, lw in METHODS:
    path = os.path.join(RUNS_DIR, f"{stem}.json")
    if not os.path.exists(path):
        missing.append(path)
        continue
    with open(path) as f:
        r = json.load(f)
    h = r.get("history", {})
    val_entries = h.get("val", [])
    val_acc = [v["acc"] for v in val_entries
               if isinstance(v, dict) and "acc" in v]
    if not val_acc:
        print(f"WARN: {path} has no val accuracy history; skipping")
        continue
    loaded.append((label, color, lw, val_acc))
    print(f"  loaded {label:<14}  epochs={len(val_acc)}  "
          f"final_val_acc={val_acc[-1]*100:.2f}%")

if missing:
    print("\nMISSING (will not appear in figure):")
    for m in missing:
        print(f"  {m}")

if not loaded:
    sys.exit("\nNo runs loaded. Are you running from ~/ViPER/notebooks ?")

# ─── Plot ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 3.0))

for label, color, lw, val_acc in loaded:
    epochs = range(1, len(val_acc) + 1)
    val_pct = [v * 100 for v in val_acc]
    ax.plot(epochs, val_pct, label=label,
            color=color, linewidth=lw, alpha=0.95)

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation accuracy (%)")
n_epochs = len(loaded[0][3])
ax.set_xlim(0, n_epochs + 1)
ax.set_ylim(60, 100)

ax.legend(loc="lower right", frameon=True, framealpha=0.92,
          edgecolor="black", borderpad=0.4, handlelength=2.0)

plt.tight_layout(pad=0.3)
plt.savefig("f2_training_curves.pdf")
plt.savefig("f2_training_curves.png", dpi=150)
plt.close()

print("\nSaved: f2_training_curves.pdf, f2_training_curves.png")
