"""Encoder comparison and world-model depth curve from summarize_runs.py output.

    python -m exp_navsim.eval.plot_encoders --summary outputs/14_summary/all_runs.csv --out outputs/07_encoders

encoders.png  final val energy, minADE@16 and dispersion per encoder / control
wm_depth.png  final val energy vs world-model block, next-frame slot and last
              observed frame, with the other encoders as horizontal lines
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ENCODERS = [("07_none", "no input"), ("07_q_rec", "q_rec"), ("02_qsem_full", "q_sem"),
            ("07_q_sem_rec", "q_sem+q_rec"), ("12_dinov2_c16", "DINOv2 (16 ch)"), ("07_dinov2", "DINOv2"),
            ("07_vjepa2", "V-JEPA 2"), ("08_wm23", "WM block 23"), ("08_wm11_ctx", "WM block 11, last frame"),
            ("07_status", "ego status"), ("07_status_cmd", "ego status + cmd")]
BLOCKS = (3, 7, 11, 15, 19, 23)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="outputs/14_summary/all_runs.csv")
    ap.add_argument("--out", default="outputs/07_encoders")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary)
    val = df[df.loader == "val"].set_index("run")

    rows = [(label, val.loc[run]) for run, label in ENCODERS if run in val.index]
    table = pd.DataFrame([{"encoder": l, "run": r.name, "val_energy": r.final_energy, "val_minade16": r.final_ade_min,
                           "val_ade_mean": r.final_ade_mean, "val_dispersion": r.final_dispersion} for l, r in rows])
    table.round(3).to_csv(out / "encoders.csv", index=False)
    print(table.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["gray" if "status" in l or l == "no input" else ("tab:green" if l.startswith("WM") else "tab:blue")
              for l in table.encoder]
    for ax, col, title in zip(axes, ("val_energy", "val_minade16", "val_dispersion"),
                              ("val energy score [m]", "val minADE@16 [m]", "val dispersion")):
        ax.barh(table.encoder, table[col], color=colors)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="x")
        if col == "val_dispersion":
            ax.axvline(1, ls="--", c="k", lw=0.8)
            ax.set_xlim(0.7, 1.05)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(out / "encoders.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for suffix, label, style in (("", "next-frame slot (predicted future)", "o-"), ("_ctx", "last observed frame", "s-")):
        pts = [(b, val.loc[f"08_wm{b}{suffix}"].final_energy) for b in BLOCKS if f"08_wm{b}{suffix}" in val.index]
        if pts:
            ax.plot(*zip(*pts), style, label=label)
    for run, label, c in (("07_vjepa2", "V-JEPA 2", "tab:purple"), ("07_dinov2", "DINOv2", "tab:orange"),
                          ("02_qsem_full", "q_sem", "tab:brown"), ("07_status", "ego status (no image)", "gray")):
        if run in val.index:
            ax.axhline(val.loc[run].final_energy, ls="--", c=c, lw=1, label=label)
    ax.set_xlabel("world-model block (0-based, of 24)")
    ax.set_ylabel("val energy score [m] (lower is better)")
    ax.set_xticks(BLOCKS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out.parent / "08_wm_blocks" / "wm_depth.png", dpi=150)
    print(f"-> {out / 'encoders.png'}, {out.parent / '08_wm_blocks' / 'wm_depth.png'}")


if __name__ == "__main__":
    main()
