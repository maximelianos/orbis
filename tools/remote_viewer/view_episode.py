#!/usr/bin/env python3
"""Watch a NAVSIM long episode as video, streamed from a remote Jupyter kernel.

Runs on your LOCAL PC. No notebook involved: it drives a kernel on the cluster
through the Jupyter server's websocket API and shows the frames in a Tk window.

Each frame is the surround-camera strip above the NAVSIM map BEV (lanes +
agents), with the ground-truth path and — when --ckpt is given — the model's
sampled trajectory distribution drawn on it, exactly as in test_model.py. A
marker walks the GT path so you can see where in the episode you are.

SETUP
-----
1. On the cluster, start a Jupyter server (in the navsim env) and note its token:

       jupyter lab --no-browser --port 8888
       jupyter server list          # -> http://localhost:8888/?token=abc123...

2. On the local PC, forward that one port (the only tunnel needed — the websocket
   API multiplexes every kernel channel over it):

       ssh -N -L 8888:localhost:8888 <user>@<cluster>

3. On the local PC, install the dependencies and copy this directory over
   (view_episode.py, kernel_client.py and remote_episode.py must sit together —
   remote_episode.py is read as text and pushed into the kernel):

       pip install websocket-client pillow

RUN
---
       python view_episode.py --token abc123 --episode 0
       python view_episode.py --token abc123 --list            # how many episodes?

       # with predicted trajectories on the BEV
       python view_episode.py --token abc123 \\
           --ckpt logs_navsim/2026-08-21T14-45-31_config/checkpoints/last.ckpt

       # start on the long, sharply-turning episodes
       python view_episode.py --token abc123 --min-distance 100 --min-angle 30

   The token can also come from the JUPYTER_TOKEN environment variable.

KEYS
----
       space        pause / resume
       left/right   step one frame (also pauses)
       n / p        next / previous episode (within the current filter)
       home         jump to the first frame
       q / escape   quit

   Keys are ignored while you are typing in one of the number fields; press
   enter there (or click "apply") to commit.

FIELDS
------
   min dist / min |angle| / min frames  — the same three episode filters as
   visualize_dataset.py and test_model.py's EPISODE_FILTERS. Applying them makes
   the kernel scan every episode's trajectory once (slow on navtrain, cached
   afterwards), then n/p walk the matching episodes.

   fps — playback speed; the dataset itself is 2 Hz.

The first call is slow (the kernel builds the dataset index, loads the
checkpoint); everything is cached in the kernel afterwards, so reopening another
episode only costs the BEV render plus inference.
"""

import argparse
import io
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

from kernel_client import KernelClient

# Default location of the orbis checkout on the cluster; the kernel is chdir'd
# here so "exp_navsim/config.yaml" resolves as it does in the shell.
DEFAULT_REPO = "/scratch/local/velikanov/work/orbis"


class Prefetcher(threading.Thread):
    """Keeps a window of frames rendered and in memory ahead of the playhead.

    This is what makes remote playback feel like video: rendering happens on the
    cluster while the GUI plays out of a local cache, so the per-frame round trip
    never shows up as a stutter.

    It also owns the ONLY reference to the kernel. The GUI thread must never
    touch it — a single websocket cannot carry two execute_request/reply pairs at
    once — so GUI requests arrive as `commands` and come back as `results`.
    """

    def __init__(self, kernel, batch, lookahead, quality, max_width):
        super().__init__(daemon=True)
        self.kernel = kernel
        self.batch, self.lookahead = batch, lookahead
        self.quality, self.max_width = quality, max_width
        self.cache = {}                       # index -> JPEG bytes, or an error string
        self.lock = threading.Lock()
        self.cursor = 0                       # playhead; written by the GUI thread
        self.n = 0                            # frames in the open episode
        self.busy = False                     # a command is running (status line)
        self.commands = queue.Queue()         # (kind, code) from the GUI
        self.results = queue.Queue()          # (kind, payload) back to the GUI
        self.stopped = threading.Event()

    def run(self):
        while not self.stopped.is_set():
            # Commands first, so a seek or a filter change never waits behind a
            # batch of frames that is about to be thrown away anyway.
            if not self.commands.empty():
                self._run_command(self.commands.get())
                continue
            start = self._next_gap()
            if start is None:                 # window full -> nothing to do
                time.sleep(0.05)
                continue
            try:
                frames = self.kernel.fetch_frames(
                    start, min(start + self.batch, self.n), self.quality, self.max_width)
            except Exception as e:            # kernel died / tunnel dropped: keep the GUI alive
                self.results.put(("error", f"prefetch: {e}"))
                time.sleep(1.0)
                continue
            with self.lock:
                self.cache.update(frames)

    def _run_command(self, command):
        """Run one GUI request on the kernel and post the parsed reply back."""
        kind, code = command
        self.busy = True
        try:
            result = self.kernel.call_json(code)
            if kind == "open":                # a new episode invalidates everything
                with self.lock:
                    self.cache.clear()
                    self.cursor = 0
                    self.n = result["num_frames"]
            self.results.put((kind, result))
        except Exception as e:
            self.results.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            self.busy = False

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
    """Tk window: the frame, a row of number fields, and a status line."""

    def __init__(self, prefetcher, args):
        self.pre, self.args = prefetcher, args
        self.fps = args.fps
        self.episodes = [args.episode]        # eligible episodes; the filter replaces this
        self.pos = 0                          # index into self.episodes
        self.info, self.n = {}, 0
        self.idx, self.playing, self.loading = 0, True, True
        self.message = ""
        self.photo = None                     # Tk drops images that nothing references
        self.drawn = None                     # (index, w, h) actually on screen

        self.root = tk.Tk()
        self.root.title("navsim episode viewer")
        self.root.geometry("1280x900")
        self.view = tk.Label(self.root, bg="black")
        self.view.pack(fill="both", expand=True)
        self._build_controls()
        self.status = tk.Label(self.root, anchor="w", font=("TkFixedFont", 9))
        self.status.pack(fill="x")

        for key, handler in (("<space>", self._toggle),
                             ("<Left>", lambda: self._step(-1)),
                             ("<Right>", lambda: self._step(1)),
                             ("<Home>", self._rewind),
                             ("n", self._next_episode), ("p", self._prev_episode),
                             ("q", self._quit), ("<Escape>", self._quit)):
            self.root.bind(key, self._shortcut(handler))
        self.root.protocol("WM_DELETE_WINDOW", lambda: self._quit())

    def _build_controls(self):
        """The number fields: three episode filters plus playback speed."""
        args = self.args
        self.spec = (("min_distance", "min dist (m)", args.min_distance),
                     ("min_angle", "min |angle| (deg)", args.min_angle),
                     ("min_frames", "min frames", args.min_frames),
                     ("fps", "fps", args.fps))
        row = tk.Frame(self.root)
        row.pack(fill="x")
        self.fields = {}
        for name, label, default in self.spec:
            tk.Label(row, text=label).pack(side="left", padx=(8, 2))
            entry = tk.Entry(row, width=6)
            entry.insert(0, f"{default:g}")
            entry.bind("<Return>", lambda _: self._apply())
            entry.pack(side="left")
            self.fields[name] = entry
        tk.Button(row, text="apply", command=self._apply).pack(side="left", padx=10)
        tk.Label(row, text="space=pause  arrows=step  n/p=episode  q=quit",
                 fg="gray40").pack(side="right", padx=8)

    # --- keys ---------------------------------------------------------------- #
    @staticmethod
    def _shortcut(handler):
        """Wrap a key handler so keystrokes typed into the number fields are ignored.

        The bindings live on the root window, so without this every "n" typed into
        a filter field would also jump to the next episode.
        """
        def on_key(event):
            if not isinstance(getattr(event, "widget", None), tk.Entry):
                handler()
        return on_key

    def _toggle(self):
        # Resuming while parked on the last frame means "play it again".
        if not self.playing and self.idx >= self.n - 1:
            self.idx = 0
        self.playing = not self.playing

    def _step(self, delta):
        self.playing = False                  # stepping implies manual control
        self.idx = max(0, min(self.idx + delta, max(self.n - 1, 0)))

    def _rewind(self):
        self.idx = 0

    def _next_episode(self):
        self._open(self.pos + 1)

    def _prev_episode(self):
        self._open(self.pos - 1)

    def _quit(self):
        self.pre.stopped.set()
        self.root.destroy()

    # --- kernel requests ----------------------------------------------------- #
    def _open_code(self, episode):
        a = self.args
        return (f"viewer_open({a.config!r}, {episode}, {a.cameras!r}, "
                f"bev={not a.no_bev}, bev_size={a.bev_size}, num_samples={a.samples})")

    def _filter_code(self, values):
        return (f"viewer_filter({values['min_distance']}, {values['min_angle']}, "
                f"{int(values['min_frames'])}, {self.args.config!r})")

    def _open(self, pos):
        """Ask the kernel for another episode; `pos` wraps around the filtered list."""
        if not self.episodes:
            return
        self.pos = pos % len(self.episodes)
        self.loading, self.idx = True, 0
        self.message = f"opening episode {self.episodes[self.pos]}"
        self.pre.commands.put(("open", self._open_code(self.episodes[self.pos])))

    def _apply(self):
        """Commit the number fields.

        fps takes effect on the next tick; the three filters go to the kernel,
        which scans every episode's trajectory once (slow) and then filters
        instantly on the cached statistics.
        """
        try:
            values = {name: float(self.fields[name].get()) for name, _, _ in self.spec}
        except ValueError:
            self.message = "number fields must be numeric"
            return
        self.fps = max(values["fps"], 0.1)
        self.root.focus_set()                 # hand the keyboard back to the window
        self.message = "scanning episodes (slow the first time)"
        self.pre.commands.put(("filter", self._filter_code(values)))

    def _drain_results(self):
        """Apply whatever the prefetcher finished since the last tick."""
        while True:
            try:
                kind, payload = self.pre.results.get_nowait()
            except queue.Empty:
                return
            if kind == "open":
                self.info, self.n = payload, payload["num_frames"]
                self.idx, self.loading, self.playing, self.drawn = 0, False, True, None
                self.message = payload.get("note", "")
                self.root.title(f"episode {payload['episode']} — {payload['token']}")
            elif kind == "filter":
                matches = payload["episodes"]
                self.message = (f"{len(matches)}/{payload['total']} episodes match"
                                if matches else "no episode matches — keeping --episode")
                # Falling back to the current list matters on startup: without it a
                # filter that matches nothing would leave the viewer with no episode
                # open and stuck showing "working" forever.
                self.episodes = matches or self.episodes
                self._open(0)
            elif kind == "error":
                self.loading, self.message = False, payload

    # --- playback ------------------------------------------------------------ #
    def _tick(self):
        """One playback step. Advances only when the next frame is already cached,
        so a slow patch of the stream stalls the video instead of skipping it."""
        self._drain_results()
        frame = None if self.loading else self.pre.get(self.idx)
        if frame is not None:
            self._show(frame)
            if self.playing:
                nxt = self.idx + 1
                if nxt >= self.n:
                    self.idx, self.playing = (0, True) if self.args.loop else (self.n - 1, False)
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
        if self.loading or self.pre.busy:
            state = "working"
        elif buffering:
            state = "buffering"
        else:
            state = "playing" if self.playing else "paused"
        panels = ("BEV+pred" if self.info.get("predicted")
                  else "BEV" if self.info.get("bev") else "cameras only")
        bits = [f"ep {self.info.get('episode', '?')} [{self.pos + 1}/{len(self.episodes)}]",
                f"frame {self.idx + 1}/{max(self.n, 1)}",
                state, f"buf {self.pre.buffered()}", panels, f"{self.fps:g} fps"]
        text = "  |  ".join(bits)
        self.status.configure(text=f" {text}" + (f"   —   {self.message}" if self.message else ""))

    def run(self):
        # Opening --episode directly skips the (slow) dataset scan; a filter given
        # on the command line asks for it up front instead.
        args = self.args
        if args.min_distance or args.min_angle or args.min_frames:
            self.message = "scanning episodes (slow the first time)"
            self.pre.commands.put(("filter", self._filter_code(
                {"min_distance": args.min_distance, "min_angle": args.min_angle,
                 "min_frames": args.min_frames})))
        else:
            self._open(0)
        self.root.after(0, self._tick)
        self.root.mainloop()


def connect(args):
    """Open the kernel, push the remote half of the viewer, load the checkpoint."""
    kernel = KernelClient(args.url, args.token, args.kernel_id).connect()
    print(f"kernel {kernel.kernel_id}")

    # chdir so the kernel resolves "exp_navsim/config.yaml" like the shell does,
    # and put the repo on sys.path in case the kernel was started elsewhere.
    kernel.execute(
        f"import os, sys\n"
        f"os.chdir({args.repo!r})\n"
        f"sys.path.insert(0, {args.repo!r}) if {args.repo!r} not in sys.path else None\n")
    kernel.execute((Path(__file__).parent / "remote_episode.py").read_text())

    if args.ckpt:
        print(f"loading {args.ckpt} (slow the first time) ...")
        print(kernel.call_json(f"viewer_model({args.ckpt!r}, {args.config!r})"))
    return kernel


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8888", help="forwarded Jupyter server")
    ap.add_argument("--token", default=os.environ.get("JUPYTER_TOKEN", ""))
    ap.add_argument("--kernel-id", default=None,
                    help="reuse a specific kernel (default: the first running one)")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="orbis checkout on the cluster")
    ap.add_argument("--config", default="exp_navsim/config.yaml")
    ap.add_argument("--episode", type=int, default=0, help="episode to open first")
    ap.add_argument("--cameras", choices=("surround", "front"), default="surround")
    ap.add_argument("--list", action="store_true", help="print the episode count and exit")
    # Model / BEV panel.
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint whose sampled trajectories are drawn on the BEV")
    ap.add_argument("--samples", type=int, default=0,
                    help="trajectory samples per episode (0 = the model's num_val_samples)")
    ap.add_argument("--no-bev", action="store_true",
                    help="skip the map BEV (needs navsim + the nuPlan maps)")
    ap.add_argument("--bev-size", type=int, default=512, help="BEV panel size in px")
    # Initial episode filter; the same three fields are editable in the GUI.
    ap.add_argument("--min-distance", type=float, default=0.0, help="m driven")
    ap.add_argument("--min-angle", type=float, default=0.0, help="deg turned")
    ap.add_argument("--min-frames", type=int, default=0)
    # Playback.
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

    prefetcher = Prefetcher(kernel, args.batch, args.lookahead, args.quality, args.max_width)
    prefetcher.start()
    Viewer(prefetcher, args).run()
    kernel.close()


if __name__ == "__main__":
    main()
