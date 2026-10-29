"""Copy the history frames of a scene index to local disk at the encoders' resolution.

    python -m exp_navsim.data.resize_frames --split navtrain

The sensor blobs sit on network storage as 1920 x 1080 JPEGs, and every encoder
pass would read them again (clip encoders read each frame several times). This
writes each frame of the index once, resized to 288 x 512 (exactly 16:9, the
tokenizer and world-model resolution), as data/frames/<split>/<id // 1000>/<id>.jpg.
JPEG draft mode decodes at half scale directly, which is most of the speed-up.
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from exp_navsim.data.scenes import load_index

SIZE = (512, 288)                       # PIL order: (W, H)
OUT = Path("data/frames")


def frame_file(split, fid):
    return OUT / split / str(fid // 1000) / f"{fid}.jpg"


def _resize(job):
    src, dst = job
    if dst.exists():
        return
    img = Image.open(src)
    img.draft("RGB", (SIZE[0] * 2, SIZE[1] * 2))
    img = img.convert("RGB").resize(SIZE, Image.BILINEAR, reducing_gap=2.0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtrain")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    index = load_index(f"data/navsim_index/{args.split}.pkl")
    root = Path(index["sensor_root"])
    jobs = [(root / p, frame_file(args.split, i)) for i, p in enumerate(index["frame_paths"])]
    with Pool(args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(_resize, jobs, chunksize=64), total=len(jobs)):
            pass


if __name__ == "__main__":
    main()
