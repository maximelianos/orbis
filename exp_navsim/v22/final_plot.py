"""Whole-split navhard two-stage EPDMS of the v22 planners next to py123d_garage.

    python -m exp_navsim.v22.final_plot

Numbers are read from outputs/24_v22/navhard.csv (score_runs) and sweeps/*_all.csv
(sweep_rank on all 225 groups); literature from plots.LITERATURE. -> final_epdms.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp_navsim.v22.plots import LITERATURE, OUT


def main():
    nh = pd.read_csv(OUT / "navhard.csv").set_index("name")["epdms"]
    sw = pd.read_csv(OUT / "sweeps" / "24g_metric_vocab_all.csv").set_index("rule")["epdms"]
    mix3 = [sw["mix_0.03"], nh["24t_mix_s1_all"], nh["24t_mix_s2_all"]]
    ours = [
        ("flow planner, waypoints (24b)", [nh["24b_flow_all"]]),
        ("metric vocab, imitation pick", [sw["imi"]]),
        ("metric vocab, EPDMS pick (GTRS)", [sw["epdms"]]),
        ("metric vocab, mixed pick, 5k targets", mix3),
        ("metric vocab, mixed pick, 10k targets", [nh["24u_metric10k_all"]]),
    ]
    rows = [(n, float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0, "ours") for n, v in ours]
    rows += [(n, s, 0.0, "metric" if m else "lit") for n, s, m in LITERATURE]
    rows.sort(key=lambda r: r[1])
    col = {"ours": "#c9792e", "lit": "#8a8f98", "metric": "#3b6ea5"}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8, 5.6))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], xerr=[r[2] for r in rows], color=[col[r[3]] for r in rows],
            height=0.68, error_kw={"elinewidth": 1, "capsize": 2})
    for i, r in enumerate(rows):
        ax.text(r[1] + r[2] + 0.6, i, f"{r[1]:.1f}", va="center", fontsize=9)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("navhard two-stage EPDMS (NAVSIM v2.2, all 225 groups)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col["ours"], label="ours (front camera, frozen Orbis features)"),
                       Patch(color=col["lit"], label="py123d_garage table, end-to-end"),
                       Patch(color=col["metric"], label="py123d_garage table, metric-trained")],
              loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "final_epdms.png", dpi=170)
    print(f"-> {OUT / 'final_epdms.png'}")


if __name__ == "__main__":
    main()
