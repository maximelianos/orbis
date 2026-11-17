"""Rerank the top-8 navhard candidates of the v22 ensemble with the dream scorer, score with fast_epdms.

    env FAST_EPDMS_CACHE_KEEP=120 python -m exp_navsim.orbis2.rerank \
        --candidates outputs/28_inputs/candidates/navhard_near.npz \
        --dream-scores dream=outputs/28_inputs/dream/navhard_near_h2/dream_scores_ds_b11.npz,ctrl=outputs/28_inputs/dream/navhard_near_h2/dream_scores_ds_b11_ctrl.npz \
        --tag near

Rules, per scorer s, over the K = 8 candidates of each token (the 8 best by mix, dream.py order):
    <s>_w<w>      mix + w * log EPDMS_dream                 (w = 0 is the ensemble's mix_0.03 plan)
    <s>_min       log EPDMS of [min(ens, dream) of NC, DAC, DDC, TLC; ens EP, TTC, LK, HC] + 0.03 imi
    <s>_minmean   the same, with the weighted terms averaged between ensemble and dream
Every rule becomes one candidate index of a single FastEPDMS call (two-stage, all 225 groups,
the aggregation of v22/sweep_rank.py: FastEPDMS.summary). Writes
outputs/28_inputs/rerank_<tag>.csv and the per-rule plans.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from exp_navsim.planner.fast_pdm import epdms
from exp_navsim.v22.fast_epdms import COLS, FastEPDMS, all_groups, group_tokens

BETA = 0.03
OUT = Path("outputs/28_inputs")


def log_ep(sub):
    return np.log(np.clip(epdms(sub.astype(np.float64)), 1e-6, None))


def rules_for(name, c, top, d, ws):
    """-> {rule: (N,) index into the K candidates}. c: candidates npz, top (N, K), d: dream scores."""
    take = lambda a: np.take_along_axis(a, top if a.ndim == 2 else top[..., None], 1)
    mix, imi, sub = take(c["mix"]), take(c["imi"]), take(c["sub"]).astype(np.float64)
    dsub = d["sub"].astype(np.float64)
    out = {f"{name}_w{w:g}": (mix + w * d["logep"]).argmax(1) for w in ws}
    mult = np.minimum(sub[..., :4], dsub[..., :4])
    out[f"{name}_min"] = (log_ep(np.concatenate([mult, sub[..., 4:]], -1)) + BETA * imi).argmax(1)
    mean = 0.5 * (sub[..., 4:] + dsub[..., 4:])
    out[f"{name}_minmean"] = (log_ep(np.concatenate([mult, mean], -1)) + BETA * imi).argmax(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--dream-scores", required=True, help="name=path[,name=path] of dream_scorer.py predict")
    ap.add_argument("--ws", default="0,0.3,1,3")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--tag", default="near")
    args = ap.parse_args()

    groups = all_groups()
    scorer = FastEPDMS(groups, workers=args.workers)          # forks before anything heavy
    c = np.load(args.candidates)
    tokens = c["token"]
    top = np.argsort(-c["mix"], axis=1, kind="stable")[:, :args.k]
    cand = np.take_along_axis(c["traj"], top[..., None, None], 1)            # (N, K, 8, 3)
    ws = [float(w) for w in args.ws.split(",")]

    rules = {}
    for item in args.dream_scores.split(","):
        name, path = item.split("=")
        d = np.load(path)
        pos = {t: i for i, t in enumerate(d["token"].tolist())}
        have = np.array([t in pos for t in tokens])
        idx = np.array([pos.get(t, 0) for t in tokens])
        assert (d["anchors"][idx][have] == top[have]).all(), "dream candidates differ from the top-K by mix"
        dd = {k: d[k][idx] for k in ("sub", "logep")}
        # tokens without a dream keep the ensemble pick (index 0 = best mix)
        dd["logep"] = np.where(have[:, None], dd["logep"], 0.0)
        dd["sub"] = np.where(have[:, None, None], dd["sub"], 1.0)
        print(f"[rerank] {name}: dream scores for {have.sum()} / {len(tokens)} tokens", flush=True)
        rules.update(rules_for(name, c, top, dd, ws))

    names = list(rules)
    stack = np.stack([cand[np.arange(len(tokens)), rules[n]] for n in names], 1)   # (N, R, 8, 3)
    need = set(group_tokens(groups))
    assert need <= set(tokens.tolist()), "the candidates must cover every token of the groups"
    df = scorer.score({t: stack[i] for i, t in enumerate(tokens) if t in need})
    s = FastEPDMS.summary(df)
    res = []
    for k, n in enumerate(names):
        r = {"rule": n, "epdms": round(100 * s.loc[k, "score"], 2),
             "changed": round(float((rules[n] != 0).mean()), 3)}
        for col in COLS[:-1]:
            r[f"{col}_s1"] = round(100 * s.loc[k, f"{col}_stage_one"], 1)
            r[f"{col}_s2"] = round(100 * s.loc[k, f"{col}_stage_two"], 1)
        res.append(r)
    res = pd.DataFrame(res)
    res.to_csv(OUT / f"rerank_{args.tag}.csv", index=False)
    np.savez_compressed(OUT / f"rerank_{args.tag}_plans.npz", token=tokens, rules=np.array(names),
                        plans=stack.astype(np.float16), picks=np.stack([rules[n] for n in names], 1))
    short = {col: col.split("_")[0][:4] + col[-3:] for col in res.columns if col.endswith(("_s1", "_s2"))}
    print(res.rename(columns=short).to_string(index=False))
    scorer.close()


if __name__ == "__main__":
    main()
