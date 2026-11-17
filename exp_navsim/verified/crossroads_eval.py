"""Does filtering or guidance cost branches at crossroads? diverse/crossroads.py with sampler specs.

    python -m exp_navsim.verified.crossroads_eval --specs 17_full_h64_full/step0100000,...@nucleus0.8@0.5

Crossroads of the validation logs (junctions_val.pkl, 651) and, with --sets val,navtest,
of navtest (junctions_navtest.pkl; `diverse/junctions.py build --index navtest`), pooled.
Same columns as outputs/17_diversity/crossroads/crossroads.csv: branches hit by >= 1 sample and
by >= 10% of the samples, share of samples on the branch the driver took, share off the
lanes, and energy / minADE against the logged future.
-> outputs/18_verified_sampling/crossroads.csv
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from exp_navsim.data.scenes import SceneFeatureDataset
from exp_navsim.diverse.crossroads import branch_stats, crossroad_rows
from exp_navsim.diverse.junctions import OUT as JOUT
from exp_navsim.diverse.train import to_gpu
from exp_navsim.eval.metrics import sample_metrics
from exp_navsim.verified.methods import METHODS, VERIFIER_V3
from exp_navsim.verified.sampler import VERIFIER, Sampler

OUT = Path("outputs/18_verified_sampling")


def load_set(name):
    """-> name, crossroad rows, taken mask, GPU data, records."""
    with open(JOUT / f"junctions_{name}.pkl", "rb") as f:
        records = pickle.load(f)
    rows, taken = crossroad_rows(records)
    if name == "navtest":
        ds = SceneFeatureDataset("data/navsim_index/navtest.pkl", features=["wm11_ctx"], split="all",
                                 require_future=False)
    else:
        ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=["wm11_ctx"], split=name)
    return name, rows, taken, to_gpu(ds, rows), records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default=None)
    ap.add_argument("--methods", action="store_true", help="the methods of methods.py, verifier v3")
    ap.add_argument("--out", default=None, help="csv path (default: the script's csv)")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--verifier", default=VERIFIER)
    ap.add_argument("--sets", default="val", help="val and / or navtest")
    args = ap.parse_args()
    if args.methods:
        specs, labels, args.verifier = [m[2] for m in METHODS], [m[0] for m in METHODS], VERIFIER_V3
    else:
        specs = args.specs.split(",")
        labels = args.labels.split(",") if args.labels else specs

    sets = [load_set(name) for name in args.sets.split(",")]
    print("crossroads: " + ", ".join(f"{n} {len(r)} ({t.sum()} taken)" for n, r, t, _, _ in sets), flush=True)
    taken = np.concatenate([t for _, _, t, _, _ in sets])

    out = []
    path = Path(args.out or OUT / "crossroads.csv")
    for spec, label in zip(specs, labels):
        sampler = Sampler(spec, args.verifier)
        m, st = {}, {}
        for _, rows, _, data, records in sets:
            s = sampler(data, args.samples)
            for k, v in sample_metrics(s, data["future"]).items():
                m.setdefault(k, []).append(v.cpu().numpy())
            for k, v in branch_stats(s.cpu().numpy(), rows, records).items():
                st.setdefault(k, []).append(np.asarray(v, dtype=float))
        m = {k: np.concatenate(v) for k, v in m.items()}
        st = {k: np.concatenate(v) for k, v in st.items()}
        row = {"model": label}
        for sub, mask in (("all", np.ones_like(taken)), ("taken", taken)):
            row[f"{sub}/energy"] = m["energy"][mask].mean()
            row[f"{sub}/ade_min"] = m["ade_min"][mask].mean()
            for k in ("branches@1", "branches@10%", "off"):
                row[f"{sub}/{k}"] = st[k][mask].mean()
        row["taken/p_taken"] = np.nanmean(st["p_taken"][taken])
        out.append(row)
        print("  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out).round(3).to_csv(path, index=False)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
