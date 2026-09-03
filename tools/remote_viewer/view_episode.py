#!/usr/bin/env python3
"""Watch a NAVSIM long episode as video, streamed from a remote Jupyter kernel.

Runs on your LOCAL PC. No notebook involved: it drives a kernel on the cluster
through the Jupyter server's websocket API and shows the frames in a Tk window.

SETUP
-----
1. On the cluster, start a Jupyter server (in the navsim env) and note its token:

       jupyter lab --no-browser --port 8888
       jupyter server list          # -> http://localhost:8888/?token=abc123...

2. On the local PC, forward that one port (this is the only tunnel needed —
   the websocket API multiplexes every kernel channel over it):

       ssh -N -L 8888:localhost:8888 <user>@<cluster>

3. On the local PC, install the two dependencies and copy this directory over
   (view_episode.py, kernel_client.py and remote_episode.py must sit together —
   remote_episode.py is read as text and pushed into the kernel):

       pip install websocket-client pillow

RUN
---
    Set token with export JUPYTER_TOKEN=""

       python view_episode.py --episode 0
       python view_episode.py --list           # how many episodes?
       python view_episode.py --episode 12 --cameras front --fps 4

KEYS
----
       space        pause / resume
       left/right   step one frame (also pauses)
       home         jump to the first frame
       q / escape   quit

The first call is slow (the kernel builds the dataset index); it is cached in the
kernel afterwards, so reopening another episode is immediate.
"""

import argparse
import io
import os
import threading
import time
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

from kernel_client import KernelClient

# Default location of the orbis repo on the cluster; the kernel is chdir'd here so
# relative paths like "exp_navsim/config.yaml" resolve the same way as in the shell.
DEFAULT_REPO = "/scratch/local/velikanov/work/orbis"


class Prefetcher(threading.Thread):
    """Keeps a window of frames ahead of the playhead decoded and in memory.

    This is what makes remote playback feel like video: rendering happens on the
    cluster while the GUI plays out of a local cache, so the per-frame SSH round
    trip never shows up as a stutter. It owns the only reference to the kernel —
    the GUI thread must never call it, since a single websocket cannot interleave
    two execute_request/reply pairs.
    """

    def __init__(self, kernel, num_frames, batch, lookahead, quality, max_width):
        super().__init__(daemon=True)
        self.kernel, self.n = kernel, num_frames
        self.batch, self.lookahead = batch, lookahead
        self.quality, self.max_width = quality, max_width
        self.cache = {}                       # index -> JPEG bytes, or an error string
        self.lock = threading.Lock()
        self.cursor = 0                       # playhead; written by the GUI thread
        self.stopped = threading.Event()

    def run(self):
        while not self.stopped.is_set():
            start = self._next_gap()
            if start is None:                 # window is full -> nothing to do
                time.sleep(0.05)
                continue
            try:
                frames = self.kernel.fetch_frames(
                    start, min(start + self.batch, self.n), self.quality, self.max_width)
            except Exception as e:            # kernel died / tunnel dropped: keep the GUI alive
                print(f"[prefetch] {e}")
                time.sleep(1.0)
                continue
            with self.lock:
                self.cache.update(frames)

    def _next_gap(self):
        """First index in [cursor, cursor + lookahead) that is not cached yet."""
        with self.lock:
            for i in range(self.cursor, min(self.cursor + self.lookahead, self.n)):
                if i not in self.cache:
                    return i
        return None

    # --- called from the GUI thread ------------------------------------------ #
    def get(self, i):
        with self.lock:
            return self.cache.get(i)

    def set_cursor(self, i):
        with self.lock:
            self.cursor = i

    def buffered(self):
        """How many consecutive frames are ready from the playhead onwards."""
        with self.lock:
            i = self.cursor
            while i in self.cache:
                i += 1
            return i - self.cursor


class Viewer:
    """Tk window: an image label plus a status line, driven by an `after` loop."""

    def __init__(self, prefetcher, info, fps, loop):
        self.pre, self.info, self.fps, self.loop = prefetcher, info, fps, loop
        self.n = info["num_frames"]
        self.idx, self.playing = 0, True
        self.photo = None                     # Tk drops images that nothing references
        self.drawn = None                     # (index, w, h) actually on screen

        self.root = tk.Tk()
        self.root.title(f"episode {info['episode']} — {info['token']}")
        self.root.geometry(f"{min(info['width'], 1600)}x{info['height'] + 24}")
        self.view = tk.Label(self.root, bg="black")
        self.view.pack(fill="both", expand=True)
        self.status = tk.Label(self.root, anchor="w", font=("TkFixedFont", 9))
        self.status.pack(fill="x")

        for key, fn in (("<space>", self._toggle), ("<Left>", lambda e: self._step(-1)),
                        ("<Right>", lambda e: self._step(1)), ("<Home>", self._rewind),
                        ("q", self._quit), ("<Escape>", self._quit)):
            self.root.bind(key, fn)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    # --- key handlers -------------------------------------------------------- #
    def _toggle(self, _=None):
        # Resuming while parked on the last frame means "play it again".
        if not self.playing and self.idx >= self.n - 1:
            self.idx = 0
        self.playing = not self.playing

    def _step(self, delta):
        self.playing = False                  # stepping implies manual control
        self.idx = max(0, min(self.idx + delta, self.n - 1))

    def _rewind(self, _=None):
        self.idx = 0

    def _quit(self, _=None):
        self.pre.stopped.set()
        self.root.destroy()

    # --- playback ------------------------------------------------------------ #
    def _tick(self):
        """One playback step. Advances only when the next frame is already cached,
        so a slow patch of the stream stalls the video instead of skipping it."""
        frame = self.pre.get(self.idx)
        if frame is not None:
            self._show(frame)
            if self.playing:
                nxt = self.idx + 1
                if nxt >= self.n:
                    self.idx, self.playing = (0, True) if self.loop else (self.n - 1, False)
                else:
                    self.idx = nxt
        self.pre.set_cursor(self.idx)
        self._update_status(buffering=frame is None)
        self.root.after(int(1000 / self.fps), self._tick)

    def _show(self, frame):
        """Render a cached frame, scaled to fit the window (never scaled up)."""
        w, h = self.view.winfo_width(), self.view.winfo_height()
        if self.drawn == (self.idx, w, h):    # nothing changed (e.g. paused) -> skip
            return
        self.drawn = (self.idx, w, h)
        if isinstance(frame, str):            # the kernel failed on this frame
            self.view.configure(image="", text=f"frame {self.idx}\n{frame}", fg="red")
            return
        img = Image.open(io.BytesIO(frame))
        if w > 1 and h > 1:
            scale = min(w / img.width, h / img.height, 1.0)
            if scale < 1.0:
                img = img.resize((max(int(img.width * scale), 1),
                                  max(int(img.height * scale), 1)), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(img)
        self.view.configure(image=self.photo, text="")

    def _update_status(self, buffering):
        state = "buffering" if buffering else ("playing" if self.playing else "paused")
        self.status.configure(
            text=f" frame {self.idx + 1}/{self.n}  |  {state}  |  "
                 f"buffered {self.pre.buffered()}  |  {self.fps:g} fps  |  "
                 f"space=pause  arrows=step  q=quit")

    def run(self):
        self.root.after(0, self._tick)
        self.root.mainloop()


def connect(args):
    """Open the kernel and push the remote half of the viewer into it."""
    kernel = KernelClient(args.url, args.token, args.kernel_id).connect()
    print(f"kernel {kernel.kernel_id}")

    # chdir so the kernel resolves "exp_navsim/config.yaml" like the shell does,
    # and put the repo on sys.path in case the kernel was started elsewhere.
    kernel.execute(
        f"import os, sys\n"
        f"os.chdir({args.repo!r})\n"
        f"sys.path.insert(0, {args.repo!r}) if {args.repo!r} not in sys.path else None\n")
    kernel.execute((Path(__file__).parent / "remote_episode.py").read_text())
    return kernel


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8888", help="forwarded Jupyter server")
    ap.add_argument("--token", default=os.environ.get("JUPYTER_TOKEN", ""))
    ap.add_argument("--kernel-id", default=None, help="reuse a specific kernel (default: first running)")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="orbis checkout on the cluster")
    ap.add_argument("--config", default="exp_navsim/config.yaml")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--cameras", choices=("surround", "front"), default="surround")
    ap.add_argument("--list", action="store_true", help="print the episode count and exit")
    ap.add_argument("--fps", type=float, default=2.0, help="playback speed (dataset is 2 Hz)")
    ap.add_argument("--loop", action="store_true", help="restart at the end instead of pausing")
    # Transport tuning: quality/max-width trade bandwidth for sharpness, batch
    # amortises the round trip, lookahead is how far ahead the cache runs.
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality")
    ap.add_argument("--max-width", type=int, default=1280, help="downscale wider frames (0 = off)")
    ap.add_argument("--batch", type=int, default=8, help="frames per kernel request")
    ap.add_argument("--lookahead", type=int, default=64, help="frames to keep buffered ahead")
    args = ap.parse_args()

    kernel = connect(args)
    if args.list:
        print(kernel.call_json(f"viewer_list({args.config!r})"))
        return

    info = kernel.call_json(
        f"viewer_open({args.config!r}, {args.episode}, {args.cameras!r})")
    print(f"episode {info['episode']}/{info['num_episodes']} — {info['token']} — "
          f"{info['num_frames']} frames at {info['width']}x{info['height']}")

    pre = Prefetcher(kernel, info["num_frames"], args.batch, args.lookahead,
                     args.quality, args.max_width)
    pre.start()
    Viewer(pre, info, args.fps, args.loop).run()
    kernel.close()


if __name__ == "__main__":
    main()
