"""PDM sub-scores of every vocabulary trajectory, per scene (Hydra-MDP distillation targets).

    # targets on the cached navtrain subset (cache_navtrain.py)
    python -m exp_navsim.planner.score_vocab --split navtrain_sub --vocab 256
    # ceiling of the vocabulary on the navtest subset: EPDMS of the best entry per scene
    python -m exp_navsim.planner.score_vocab --split navtest --vocab 256 --subset outputs/16_fast_planner/subset_500.json

Writes outputs/16_fast_planner/vocab<V>_scores_<split>.npz: tokens, rows (navtrain
index rows, for the training script), scores (N, V, 8) in fast_pdm.METRICS order.
Two-frame extended comfort is left out: it needs the planner's previous output.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from exp_navsim.planner.fast_pdm import METRICS, FastPDM, epdms, load_subset

OUT = Path("outputs/16_fast_planner")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtrain_sub")
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--subset", default=None, help="subset json (navtest); default: navtrain_sub_tokens.json")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=480)
    args = ap.parse_args()

    if args.subset:
        tokens, _ = load_subset(args.subset)
        rows = np.full(len(tokens), -1)
    else:
        d = json.load(open(OUT / "navtrain_sub_tokens.json"))
        tokens, rows = d["tokens"], np.asarray(d["rows"])
    vocab = np.load(OUT / f"vocab_{args.vocab}.npy").astype(np.float32)
    out = OUT / f"vocab{args.vocab}_scores_{args.split}.npz"
    part = out.with_suffix(".part.npz")
    done = dict(np.load(part)) if part.exists() else {"tokens": np.array([], dtype=str),
                                                     "scores": np.zeros((0, len(vocab), len(METRICS)), np.float16)}
    have = set(done["tokens"].tolist())
    todo = [t for t in tokens if t not in have]
    print(f"{len(tokens)} scenes, {len(have)} done, vocab {len(vocab)}", flush=True)

    scorer = FastPDM(args.split, todo, workers=args.workers, preload=False)
    toks, scores = list(done["tokens"]), [done["scores"]]
    t0 = time.time()
    for i in range(0, len(todo), args.chunk):
        chunk = todo[i:i + args.chunk]
        res = scorer.candidate_scores({t: vocab for t in chunk})
        toks += chunk
        scores.append(np.stack([res[t] for t in chunk]).astype(np.float16))
        np.savez(part, tokens=np.array(toks), scores=np.concatenate(scores))
        rate = (i + len(chunk)) / (time.time() - t0)
        print(f"{i + len(chunk)}/{len(todo)}  {rate:.1f} scenes/s  eta {(len(todo) - i - len(chunk)) / rate / 60:.0f} min",
              flush=True)
    scorer.close()

    S = np.concatenate(scores)
    pos = {t: j for j, t in enumerate(toks)}
    order = [pos[t] for t in tokens]
    S = S[order]
    np.savez(out, tokens=np.array(tokens), rows=rows, scores=S)
    part.unlink(missing_ok=True)
    e = epdms(S.astype(np.float64))                                  # (N, V), without two-frame comfort
    print(f"-> {out}")
    print(f"EPDMS without two-frame comfort: best entry per scene {e.max(1).mean():.4f}, "
          f"mean over entries {e.mean():.4f}, best single entry for all scenes {e.mean(0).max():.4f}")
    for j, m in enumerate(METRICS):
        print(f"  {m:32s} fraction of entries passing / mean {S[..., j].astype(np.float32).mean():.3f}")


if __name__ == "__main__":
    main()
