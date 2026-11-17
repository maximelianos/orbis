"""PDM sub-score targets of our anchor vocabulary for the SimScale scenes, from the released rewards.

    # 1. stream the 19 GB reward pickle once (file or URL), keep our 256 anchors -> data/simscale/rewards/
    python -m exp_navsim.simscale.rewards extract --round pdm_v1.0-0 --vocab 256
    # 2. rows of a scene index -> npz in the format of train.py --score-targets (+ dense rows file)
    python -m exp_navsim.simscale.rewards align --round pdm_v1.0-0 --vocab 256 --split simscale_pdm0

The same extract pass also keeps a dense subset of the GTRS vocabulary (--dense 2048, GTRS-Dense
style training of the scorer): the 256 GTRS entries our anchors map to plus 1792 entries chosen by
farthest-point sampling among the plausible ones (dense_index). Their scores go to
data/simscale/rewards/<round>_dense2048.npy, float16 (N, 2048, 8) in METRICS order, a plain .npy
that np.load(..., mmap_mode="r") maps, rows in the order of the tokens of <round>_vocab256.npz.

SimScale_rewards/simscale_16384_<round>.pkl is one dict {scene token: {metric: float16 (16384,)}}
with the PDM sub-scores of the GTRS vocabulary traj_final/16384.npy (16384 x 40 x 3, 10 Hz, 4 s)
for every sim scene; SimScale's GTRS agent pickle.loads the whole file. Here it is read with a
pure-Python unpickler that hands every top-level item to a callback and drops it, so the peak
memory is about one SETITEMS batch (1000 scenes, ~0.3 GB) instead of the whole file.

Each of our anchors (outputs/16_fast_planner/vocab_<V>.npy, 8 poses at 2 Hz) takes the scores of
its nearest GTRS trajectory (mean x, y distance over the 8 poses at 0.5 .. 4 s). The output keeps
the order of planner/fast_pdm.METRICS.
"""

import argparse
import io
import json
import pickle
import resource
import time
from pathlib import Path

import numpy as np

from exp_navsim.planner.fast_pdm import METRICS
from exp_navsim.simscale.safe_pickle import find_class

HF = "https://huggingface.co/datasets/OpenDriveLab/SimScale/resolve/main/SimScale_rewards"
GTRS_VOCAB = Path("data/simscale/gtrs_vocab_16384.npy")
OUR_VOCAB = Path("outputs/16_fast_planner")
OUT = Path("data/simscale/rewards")
NAVTRAIN = Path("data/navsim_index/navtrain.pkl")

_BIG = object()     # memo placeholder for objects we do not keep


def _small(v):
    if isinstance(v, (str, int, float, bool, type(None), type, np.dtype)) or callable(v):
        return True
    if isinstance(v, (bytes, bytearray)):
        return len(v) <= 1024
    if isinstance(v, tuple):
        return all(_small(x) for x in v)
    return False


class StreamingUnpickler(pickle._Unpickler):
    """Unpickles a top-level dict item by item; on_item(key, value) gets each item once."""

    dispatch = dict(pickle._Unpickler.dispatch)

    def __init__(self, file, on_item):
        super().__init__(file)
        self.on_item = on_item
        self.root = None

    def find_class(self, module, name):
        return find_class(module, name)

    def load_empty_dictionary(self):
        d = {}
        if self.root is None:
            self.root = d
        self.append(d)
    dispatch[pickle.EMPTY_DICT[0]] = load_empty_dictionary

    def _drop_big(self):
        # Called once the top-level items of a batch are handed over: the memo would otherwise
        # pin every scene. Strings, numbers, classes, dtypes, short bytes and tuples of those stay
        # (BINGET reuses them across scenes); arrays, dicts and tuples holding array buffers are
        # only referenced inside their own scene.
        for k, v in self.memo.items():
            if v is not _BIG and not _small(v):
                self.memo[k] = _BIG

    def _get(self, i):
        v = self.memo[i]
        if v is _BIG:
            raise pickle.UnpicklingError(f"memo {i} refers to a dropped object")
        self.append(v)

    def load_binget(self):
        self._get(self.read(1)[0])
    dispatch[pickle.BINGET[0]] = load_binget

    def load_long_binget(self):
        self._get(int.from_bytes(self.read(4), "little"))
    dispatch[pickle.LONG_BINGET[0]] = load_long_binget

    def load_setitems(self):
        items = self.pop_mark()
        d = self.stack[-1]
        if d is not self.root:
            for i in range(0, len(items), 2):
                d[items[i]] = items[i + 1]
            return
        for i in range(0, len(items), 2):
            self.on_item(items[i], items[i + 1])
        self._drop_big()
    dispatch[pickle.SETITEMS[0]] = load_setitems

    def load_setitem(self):
        value, key = self.stack.pop(), self.stack.pop()
        d = self.stack[-1]
        if d is self.root:
            self.on_item(key, value)
            self._drop_big()
        else:
            d[key] = value
    dispatch[pickle.SETITEM[0]] = load_setitem


def open_source(src):
    if src.startswith("http"):
        import requests
        resp = requests.get(src, stream=True, timeout=60)
        resp.raise_for_status()
        return io.BufferedReader(resp.raw, 8 << 20)
    return open(src, "rb", buffering=8 << 20)


def gtrs_vocab():
    if not GTRS_VOCAB.exists():
        import urllib.request
        GTRS_VOCAB.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/OpenDriveLab/SimScale/raw/main/traj_final/16384.npy", GTRS_VOCAB)
    return np.load(GTRS_VOCAB, allow_pickle=False)          # (16384, 40, 3), 10 Hz


def nearest_gtrs(vocab):
    """Our (V, 8, 3) anchors -> index of the nearest GTRS entry and the mean pose distance (m)."""
    g = gtrs_vocab()[:, 4::5, :2]                            # 0.5 .. 4.0 s, as our 8 poses
    d = np.linalg.norm(vocab[:, None, :, :2] - g[None], axis=-1).mean(-1)
    return d.argmin(1), d.min(1)


def dense_paths(n):
    return OUT / f"dense{n}_gtrs_index.npy", OUT / f"dense{n}_traj.npy"


def logged_futures():
    """(N, 8, 2) logged navtrain futures (x, y), the scenes with a full future."""
    with open(NAVTRAIN, "rb") as f:
        fut = np.asarray(pickle.load(f)["future"], np.float32)
    return fut[np.isfinite(fut).all((1, 2))][..., :2]


def nearest_logged(traj, logged, chunk=64):
    """Mean x, y pose distance (m) of every (K, 8, 2+) trajectory to its nearest logged future."""
    import torch
    F = torch.from_numpy(logged).flatten(1)
    out = []
    for c in torch.from_numpy(np.ascontiguousarray(traj[..., :2], np.float32)).flatten(1).split(chunk):
        out.append((c[:, None] - F[None]).unflatten(-1, (8, 2)).norm(dim=-1).mean(-1).min(1).values)
    return torch.cat(out).numpy()


def make_dense_index(n, vocab=256):
    """The n GTRS indices of the dense scorer vocabulary, deterministic.

    First the GTRS neighbours of our anchors (nearest_gtrs, vocab order), then farthest-point
    sampling (mean x, y pose distance, the metric of nearest_gtrs) over the plausible GTRS
    entries. Plausible: end point no farther than the farthest logged navtrain end point, and
    the nearest logged navtrain future within the largest such distance of our own anchors
    (the GTRS vocabulary was clustered from human driving, so the end point rule alone drops
    nothing; the second rule drops the few entries that are farther from any logged drive than
    any of our anchors is)."""
    seeds, _ = nearest_gtrs(np.load(OUR_VOCAB / f"vocab_{vocab}.npy"))
    assert len(set(seeds.tolist())) == len(seeds), "two anchors share a GTRS entry"
    ref = OUT / f"pdm_v1.0-0_vocab{vocab}.npz"
    if ref.exists():
        assert np.array_equal(np.load(ref)["gtrs_index"], seeds), f"seeds differ from {ref}"
    g = gtrs_vocab()[:, 4::5]                                         # (16384, 8, 3)
    logged = logged_futures()
    end_max = float(np.linalg.norm(logged[:, -1], axis=-1).max())
    log_d = nearest_logged(g, logged)
    anchor_max = float(nearest_logged(np.load(OUR_VOCAB / f"vocab_{vocab}.npy"), logged).max())
    ok = (np.linalg.norm(g[:, -1, :2], axis=-1) <= end_max) & (log_d <= anchor_max)
    ok[seeds] = True
    print(f"[dense] plausible GTRS entries {ok.sum()} / {len(g)} (end point <= {end_max:.1f} m: "
          f"{(np.linalg.norm(g[:, -1, :2], axis=-1) <= end_max).sum()}, nearest logged future <= "
          f"{anchor_max:.2f} m: {(log_d <= anchor_max).sum()})", flush=True)
    xy = g[..., :2].astype(np.float64)
    dmin = np.full(len(g), np.inf)
    for i in seeds:
        dmin = np.minimum(dmin, np.linalg.norm(xy - xy[i], axis=-1).mean(-1))
    chosen = list(seeds)
    while len(chosen) < n:
        i = int(np.argmax(np.where(ok, dmin, -1.0)))
        chosen.append(i)
        dmin = np.minimum(dmin, np.linalg.norm(xy - xy[i], axis=-1).mean(-1))
    idx = np.array(chosen, dtype=np.int64)
    print(f"[dense] {n} entries; after sampling, plausible entries are within {dmin[ok].max():.3f} m "
          f"(mean {dmin[ok].mean():.3f} m) of the set", flush=True)
    return idx, g[idx].astype(np.float32)


def dense_index(n):
    """(GTRS indices (n,), trajectories (n, 8, 3)) of the dense vocabulary; made once, then loaded."""
    ip, tp = dense_paths(n)
    if not ip.exists():
        idx, traj = make_dense_index(n)
        ip.parent.mkdir(parents=True, exist_ok=True)
        np.save(tp, traj)
        np.save(ip, idx)
        print(f"[dense] -> {ip}, {tp}", flush=True)
    return np.load(ip), np.load(tp)


def npy_header(shape, dtype=np.float16):
    b = io.BytesIO()
    np.lib.format.write_array_header_1_0(b, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
                                             "fortran_order": False, "shape": tuple(shape)})
    return b.getvalue()


def cmd_dense_index(args):
    idx, traj = dense_index(args.dense)
    print(f"[dense] {len(idx)} GTRS indices, trajectories {traj.shape}")


def cmd_extract(args):
    vocab = np.load(OUR_VOCAB / f"vocab_{args.vocab}.npy")
    nn, dist = nearest_gtrs(vocab)
    print(f"[nn] mean pose L2 to the nearest GTRS entry: median {np.median(dist):.3f}  mean {dist.mean():.3f}  "
          f"p90 {np.percentile(dist, 90):.3f}  max {dist.max():.3f} m; {len(set(nn))} distinct entries", flush=True)
    src = args.src or f"{HF}/simscale_16384_{args.round}.pkl"
    tokens, scores, pdms, gt, keys, t0 = [], [], [], [], {}, time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dense_f = None
    if args.dense:
        # streamed straight to a .npy: the header is written for a placeholder row count and
        # rewritten at the end (both 128 bytes), so no copy of the scores is held in memory
        didx, _ = dense_index(args.dense)
        dense_out = OUT / f"{args.round}_dense{args.dense}.npy"
        dense_part = dense_out.with_suffix(".npy.part")
        head = npy_header((10 ** 7, args.dense, len(METRICS)))
        dense_f = open(dense_part, "wb", buffering=16 << 20)
        dense_f.write(head)

    def on_item(token, value):
        tokens.append(token)
        for k in value:
            keys[k] = keys.get(k, 0) + 1
        scores.append(np.stack([np.asarray(value[m])[nn] for m in METRICS], -1).astype(np.float16))
        pdms.append(np.asarray(value["pdm_score"])[nn].astype(np.float16))
        gt.append(np.asarray(value["near_human_gt_trajectory"], dtype=np.float32)[4::5])
        if dense_f is not None:
            dense_f.write(np.stack([np.asarray(value[m])[didx] for m in METRICS], -1)
                          .astype(np.float16).tobytes())
        if len(tokens) % 5000 == 0:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            print(f"[extract] {len(tokens)} scenes  {time.time() - t0:.0f}s  peak RSS {rss:.2f} GB", flush=True)

    with open_source(src) as f:
        StreamingUnpickler(f, on_item).load()
    S = np.stack(scores)
    if dense_f is not None:
        real = npy_header((len(tokens), args.dense, len(METRICS)))
        assert len(real) == len(head)
        dense_f.seek(0)
        dense_f.write(real)
        dense_f.close()
        dense_part.rename(dense_out)
        np.save(OUT / f"{args.round}_dense{args.dense}_tokens.npy", np.array(tokens))
        D = np.load(dense_out, mmap_mode="r")
        assert D.shape[0] == len(tokens)
        assert np.array_equal(D[:, :args.vocab], S), "dense columns 0..255 differ from the vocab scores"
        print(f"[extract] -> {dense_out} {D.shape} float16, {dense_out.stat().st_size / 1e9:.2f} GB", flush=True)
    out = OUT / f"{args.round}_vocab{args.vocab}.npz"
    np.savez(out, tokens=np.array(tokens), scores=S, pdm_score=np.stack(pdms), gt_traj=np.stack(gt),
             gtrs_index=nn, gtrs_dist=dist)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[extract] {len(tokens)} scenes, metric keys {keys}")
    print(f"[extract] -> {out} {S.shape}  {time.time() - t0:.0f}s  peak RSS {rss:.2f} GB")
    print("[extract] mean sub-scores:", dict(zip(METRICS, np.round(S.astype(np.float32).mean((0, 1)), 3))))


def cmd_align(args):
    d = np.load(OUT / f"{args.round}_vocab{args.vocab}.npz")
    with open(Path("data/navsim_index") / f"{args.split}.pkl", "rb") as f:
        index = pickle.load(f)
    row_of = {t: i for i, t in enumerate(index["token"])}
    have = np.array([t in row_of for t in d["tokens"]])
    rows = np.array([row_of[t] for t in d["tokens"][have]], dtype=np.int64)
    order = np.argsort(rows)
    out = Path(args.out or f"outputs/28_inputs/vocab{args.vocab}_scores_{args.split}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, tokens=d["tokens"][have][order], rows=rows[order], scores=d["scores"][have][order])
    print(f"[align] reward scenes {len(have)}, in index {have.sum()}, index scenes {len(index['token'])}, "
          f"index scenes without rewards {len(index['token']) - have.sum()} -> {out}")
    dense = OUT / f"{args.round}_dense{args.dense}.npy"
    if args.dense and dense.exists():
        # no copy of the 2 GB scores: rows of the split index -> src rows of the round's .npy
        assert np.array_equal(np.load(OUT / f"{args.round}_dense{args.dense}_tokens.npy"), d["tokens"])
        src = np.flatnonzero(have)[order]
        dout = OUT / f"dense{args.dense}_{args.split}.npz"
        np.savez(dout, tokens=d["tokens"][have][order], rows=rows[order], src=src,
                 scores_file=str(dense), gtrs_index=str(dense_paths(args.dense)[0]))
        print(f"[align] dense rows -> {dout} (scores in {dense})")
    elif args.dense:
        print(f"[align] no {dense}, dense rows skipped")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "align", "dense-index"):
        p = sub.add_parser(name)
        p.add_argument("--round", default="pdm_v1.0-0")
        p.add_argument("--vocab", type=int, default=256)
        p.add_argument("--dense", type=int, default=2048, help="dense GTRS subset size; 0 = off")
        if name == "extract":
            p.add_argument("--src", default=None, help="local pkl path or URL (default: HF URL)")
        else:
            p.add_argument("--split", default="simscale_pdm0")
            p.add_argument("--out", default=None)
    args = ap.parse_args()
    {"extract": cmd_extract, "align": cmd_align, "dense-index": cmd_dense_index}[args.cmd](args)


if __name__ == "__main__":
    main()
