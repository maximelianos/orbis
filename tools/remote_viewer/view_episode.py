#!/usr/bin/env python3
"""Watch NAVSIM long episodes as video, streamed from a remote Jupyter kernel.

Runs on your LOCAL PC. No notebook involved: it drives a kernel on the cluster
through the Jupyter server's websocket API and shows the frames in a Tk window.

Each frame is the front camera above the NAVSIM map BEV (lanes + agents). On the
BEV: the whole ground-truth path, the window the model is being asked about, and
— with --ckpt — the trajectory distribution the model predicts FROM THAT FRAME,
rotated into the ego heading there. The prediction is re-run at every frame, so
playing the episode shows the forecast evolving. Near the end of the episode the
window runs past the last recorded pose: the model still predicts a future there,
it simply has no ground truth to be compared against.

Every knob lives in the SETTINGS dict below; the command line just overrides it.

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
       python view_episode.py --token abc123

       # with the model's predicted trajectories on the BEV
       python view_episode.py --token abc123 \\
           --ckpt logs_navsim/2026-08-21T14-45-31_config/checkpoints/last.ckpt

   The token can also come from the JUPYTER_TOKEN environment variable.

KEYS
----
       space        pause / resume
       left/right   step one frame (also pauses)
       n / p        next / previous episode
       home         jump to the first frame
       q / escape   quit

   Keys are ignored while you are typing in one of the number fields; press
   enter there (or click "apply") to commit.

BUFFERING
---------
The worker thread buffers EVERY frame of the current episode and of the
`neighbours` episodes either side, in priority order (current, +1, -1, +2, -2,
...). Nothing the GUI does waits on the network: n/p switch instantly once a
neighbour is buffered, and the kernel keeps those episodes' rasterised BEVs warm
so re-opening one costs no render either.

NOTE ON THE SPLIT
-----------------
The long dataloader runs with split "all" while data.params.validation is split
"val", so by default only ~10% of browsable episodes would have cached latents to
predict from. pred_split defaults to "all" so every cached episode can be
predicted; the status line then shows whether the open episode is train or val,
because a prediction on a train episode was fitted on that very trajectory. Set
pred_split to "val" to restrict predictions to genuinely unseen episodes.
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

# --------------------------------------------------------------------------- #
# Every tunable. The command line overrides these; the GUI number fields write
# back into them, so this dict is the single source of truth at runtime.
# --------------------------------------------------------------------------- #
SETTINGS = {
    # --- connection ---
    "url": "http://localhost:8888",
    "token": os.environ.get("JUPYTER_TOKEN", ""),
    "kernel_id": None,                  # None = reuse the first running kernel
    "repo": "/scratch/local/velikanov/work/orbis",
    "config": "exp_navsim/config.yaml",

    # --- episode filter (same criteria as test_model.py's EPISODE_FILTERS) ---
    "min_distance": 20.0,               # m driven
    "min_angle": 50.0,                  # deg between initial heading and start->end
    "min_frames": 20,                   # frames in the episode

    # --- model ---
    "ckpt": None,
    "pred_split": "all",                # which cached episodes may be predicted
    "total_len": 20,                    # model window: context + predicted frames
    "samples": 0,                       # 0 = the model's num_val_samples

    # --- rendering ---
    "cameras": "front",                 # "front" | "surround"
    "bev": True,
    "bev_size": 512,                    # BEV panel render size in px
    "width": 768,                       # final frame width in px (0 = as composited)
    "quality": 85,                      # JPEG quality

    # --- playback / buffering ---
    "fps": 2.0,                         # the dataset itself is 2 Hz
    "loop": False,
    "neighbours": 3,                    # episodes buffered either side of the current
    "batch": 8,                         # frames per kernel request
}

# Options the kernel needs for rendering; sent once by connect().
_REMOTE_OPTS = ("config", "cameras", "bev", "bev_size", "width", "quality")


class Prefetcher(threading.Thread):
    """Buffers whole episodes in the background; the GUI never waits on it.

    Work is driven by a `plan`: a list of episode ids in priority order (the one
    on screen first, then its neighbours outwards). The thread walks the plan,
    fetching each episode's geometry and then every one of its frames, so by the
    time you press n the next episode is usually already complete.

    It also owns the ONLY reference to the kernel — a single websocket cannot
    carry two execute_request/reply pairs at once — so GUI requests arrive as
    `commands` and come back as `results`.
    """

    def __init__(self, kernel, settings):
        super().__init__(daemon=True)
        self.kernel, self.settings = kernel, settings
        self.cache = {}                  # (episode, frame) -> JPEG bytes | error str
        self.infos = {}                  # episode -> geometry dict (small, kept forever)
        self.plan = []                   # episode ids, highest priority first
        self.lock = threading.Lock()
        self.total_len = settings["total_len"]
        # Bumped whenever cached frames stop being valid (a new model window).
        # A fetch that started before the bump must not commit its frames.
        self.version = 0
        self.busy = ""                   # name of the running command, for the status
        self.commands = queue.Queue()    # (kind, code) from the GUI
        self.results = queue.Queue()     # (kind, payload) back to the GUI
        self.stopped = threading.Event()

    # --- GUI-facing ---------------------------------------------------------- #
    def set_plan(self, episodes):
        """Replace the buffering plan and drop frames outside it."""
        with self.lock:
            self.plan = list(episodes)
            keep = set(self.plan)
            for key in [k for k in self.cache if k[0] not in keep]:
                del self.cache[key]

    def set_total_len(self, total_len):
        """Change the model window; every cached frame was drawn with the old one."""
        with self.lock:
            if total_len != self.total_len:
                self.total_len = total_len
                self.cache.clear()
                self.version += 1

    def get(self, episode, frame):
        with self.lock:
            return self.cache.get((episode, frame))

    def progress(self, episode):
        """(frames buffered here, frames total, episodes complete, episodes planned)."""
        with self.lock:
            info = self.infos.get(episode)
            total = info["num_frames"] if info else 0
            here = sum((episode, i) in self.cache for i in range(total))
            done = 0
            for ep in self.plan:
                other = self.infos.get(ep)
                if other and all((ep, i) in self.cache
                                 for i in range(other["num_frames"])):
                    done += 1
            return here, total, done, len(self.plan)

    # --- worker -------------------------------------------------------------- #
    def run(self):
        while not self.stopped.is_set():
            # Commands first: a filter change or a checkpoint load must not wait
            # behind a queue of frames.
            if not self.commands.empty():
                self._run_command(self.commands.get())
                continue
            job = self._next_job()
            if job is None:
                time.sleep(0.05)
                continue
            self._run_job(job)

    def _run_command(self, command):
        """Run one GUI request on the kernel and post the parsed reply back."""
        kind, code = command
        self.busy = kind
        try:
            self.results.put((kind, self.kernel.call_json(code)))
        except Exception as e:
            self.results.put(("error", f"{kind}: {type(e).__name__}: {e}"))
        finally:
            self.busy = ""

    def _next_job(self):
        """The most valuable outstanding piece of work, or None if fully buffered.

        Walks the plan in order, so the episode on screen is always finished
        before a neighbour is started.
        """
        with self.lock:
            plan, version, total_len = list(self.plan), self.version, self.total_len
            for episode in plan:
                info = self.infos.get(episode)
                if info is None:
                    return ("info", episode, 0, version, total_len)
                for i in range(info["num_frames"]):
                    if (episode, i) not in self.cache:
                        return ("frames", episode, i, version, total_len)
        return None

    def _run_job(self, job):
        kind, episode, start, version, total_len = job
        try:
            if kind == "info":
                info = self.kernel.call_json(f"viewer_prepare({episode})")
                with self.lock:
                    self.infos[episode] = info
                self.results.put(("prepared", info))
                return
            stop = start + self.settings["batch"]
            frames, note = self.kernel.fetch_frames(
                episode, start, stop, total_len, self.settings["samples"])
            if note:
                self.results.put(("error", note))
        except Exception as e:           # kernel died / tunnel dropped: stay alive
            self.results.put(("error", f"buffering: {type(e).__name__}: {e}"))
            time.sleep(1.0)
            return
        with self.lock:
            if version == self.version:  # nothing invalidated these meanwhile
                got = {(episode, i): data for i, data in frames.items()}
                # Never leave a hole the planner would pick again next pass: a
                # frame the kernel did not return is recorded as an error, or the
                # same span would be refetched forever.
                info = self.infos.get(episode)
                limit = min(stop, info["num_frames"] if info else stop)
                for i in range(start, limit):
                    got.setdefault((episode, i), "no frame returned")
                self.cache.update(got)


class Viewer:
    """Tk window: the frame, a row of number fields, and a status line."""

    def __init__(self, prefetcher, settings):
        self.pre, self.settings = prefetcher, settings
        self.episodes = []                    # dataset episode ids passing the filter
        self.pos = 0                          # index into self.episodes
        self.n = 0
        self.idx, self.playing = 0, True
        self.message = "scanning episodes (slow the first time)"
        self.photo = None                     # Tk drops images nothing references
        self.drawn = None                     # (episode, index, w, h) on screen

        self.root = tk.Tk()
        self.root.title("navsim episode viewer")
        self.root.geometry("900x1000")
        self.view = tk.Label(self.root, bg="black")
        self.view.pack(fill="both", expand=True)
        self._build_controls()
        self.status = tk.Label(self.root, anchor="w", font=("TkFixedFont", 9))
        self.status.pack(fill="x")

        for key, handler in (("<space>", self._toggle),
                             ("<Left>", lambda: self._step(-1)),
                             ("<Right>", lambda: self._step(1)),
                             ("<Home>", self._rewind),
                             ("n", lambda: self._open(self.pos + 1)),
                             ("p", lambda: self._open(self.pos - 1)),
                             ("q", self._quit), ("<Escape>", self._quit)):
            self.root.bind(key, self._shortcut(handler))
        self.root.protocol("WM_DELETE_WINDOW", lambda: self._quit())

    def _build_controls(self):
        """Number fields, each bound to its SETTINGS key."""
        self.spec = (("total_len", "model len"), ("fps", "fps"),
                     ("min_distance", "min dist (m)"), ("min_angle", "min |angle| (deg)"),
                     ("min_frames", "min frames"))
        row = tk.Frame(self.root)
        row.pack(fill="x")
        self.fields = {}
        for name, label in self.spec:
            tk.Label(row, text=label).pack(side="left", padx=(8, 2))
            entry = tk.Entry(row, width=6)
            entry.insert(0, f"{self.settings[name]:g}")
            entry.bind("<Return>", lambda _: self._apply())
            entry.pack(side="left")
            self.fields[name] = entry
        tk.Button(row, text="apply", command=self._apply).pack(side="left", padx=10)
        tk.Label(row, text="space=pause  arrows=step  n/p=episode  q=quit",
                 fg="gray40").pack(side="right", padx=8)

    @property
    def episode(self):
        return self.episodes[self.pos] if self.episodes else None

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

    def _quit(self):
        self.pre.stopped.set()
        self.root.destroy()

    # --- episode navigation -------------------------------------------------- #
    def _plan(self):
        """Episodes to keep buffered: current first, then neighbours outwards.

        That order is also the fetch order, so the episode on screen is always
        finished before anything is spent on a neighbour.
        """
        if not self.episodes:
            return []
        offsets = [0]
        for d in range(1, self.settings["neighbours"] + 1):
            offsets += [d, -d]
        ids, seen = [], set()
        for off in offsets:
            episode = self.episodes[(self.pos + off) % len(self.episodes)]
            if episode not in seen:
                seen.add(episode)
                ids.append(episode)
        return ids

    def _open(self, pos):
        """Switch episode. Purely local — no round trip, so it is instant."""
        if not self.episodes:
            return
        self.pos = pos % len(self.episodes)
        self.idx, self.playing, self.drawn = 0, True, None
        self.pre.set_plan(self._plan())

    def _apply(self):
        """Commit the number fields into SETTINGS.

        fps takes effect on the next tick and the model window on the next fetch
        (it invalidates every buffered frame, which were drawn with the old one).
        The three filters only go to the kernel when they actually changed —
        re-sending them would trigger the full dataset scan for nothing.
        """
        try:
            values = {name: float(self.fields[name].get()) for name, _ in self.spec}
        except ValueError:
            self.message = "number fields must be numeric"
            return
        before = [self.settings[k] for k in ("min_distance", "min_angle", "min_frames")]
        self.settings.update(fps=max(values["fps"], 0.1),
                             total_len=max(int(values["total_len"]), 1),
                             min_distance=values["min_distance"],
                             min_angle=values["min_angle"],
                             min_frames=int(values["min_frames"]))
        self.pre.set_total_len(self.settings["total_len"])
        self.drawn = None                     # force a redraw with the new window
        self.root.focus_set()                 # hand the keyboard back to the window

        after = [self.settings[k] for k in ("min_distance", "min_angle", "min_frames")]
        if after != before:
            self.message = "scanning episodes (slow the first time)"
            self.pre.commands.put(("filter", filter_code(self.settings)))

    def _drain_results(self):
        """Apply whatever the worker finished since the last tick."""
        while True:
            try:
                kind, payload = self.pre.results.get_nowait()
            except queue.Empty:
                return
            if kind == "filter":
                matches = payload["episodes"]
                self.message = (f"{len(matches)}/{payload['total']} episodes match"
                                if matches else "no episode matches the filter")
                if matches:
                    self.episodes, self.pos = matches, 0
                    self._open(0)
            elif kind == "prepared":
                if payload["episode"] == self.episode:
                    self.message = payload.get("note", "")
            elif kind == "model":
                self.message = f"model on {payload['device']}, {payload['split']} split"
            elif kind == "error":
                self.message = payload

    # --- playback ------------------------------------------------------------ #
    def _tick(self):
        """One playback step. Advances only when the next frame is already buffered,
        so a slow patch of the stream stalls the video instead of skipping it."""
        self._drain_results()
        episode = self.episode
        info = self.pre.infos.get(episode) if episode is not None else None
        self.n = info["num_frames"] if info else 0

        frame = self.pre.get(episode, self.idx) if info else None
        if frame is not None:
            self._show(episode, frame)
            if self.playing:
                nxt = self.idx + 1
                if nxt >= self.n:
                    self.idx, self.playing = ((0, True) if self.settings["loop"]
                                              else (self.n - 1, False))
                else:
                    self.idx = nxt
        self._update_status(info, buffering=frame is None)
        self.root.after(int(1000 / self.settings["fps"]), self._tick)

    def _show(self, episode, frame):
        """Render a buffered frame, scaled to fit the window (never scaled up)."""
        w, h = self.view.winfo_width(), self.view.winfo_height()
        if self.drawn == (episode, self.idx, w, h):   # nothing changed -> skip
            return
        self.drawn = (episode, self.idx, w, h)
        if isinstance(frame, str):                    # the kernel failed on this frame
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

    def _update_status(self, info, buffering):
        here, total, done, planned = self.pre.progress(self.episode)
        if self.pre.busy:
            state = self.pre.busy
        elif buffering:
            state = "buffering"
        else:
            state = "playing" if self.playing else "paused"
        panels = ("BEV+pred" if (info or {}).get("predicted")
                  else "BEV" if (info or {}).get("bev") else "cameras only")
        if (info or {}).get("predicted"):
            panels += f" ({info.get('split', '?')})"
        bits = [f"ep {self.episode} [{self.pos + 1}/{len(self.episodes)}]",
                f"frame {self.idx + 1}/{max(self.n, 1)}", state,
                f"buf {here}/{max(total, 1)} eps {done}/{max(planned, 1)}",
                panels, f"len {self.settings['total_len']}",
                f"{self.settings['fps']:g} fps"]
        self.status.configure(text=" " + "  |  ".join(bits)
                              + (f"   —   {self.message}" if self.message else ""))

    def run(self):
        self.root.after(0, self._tick)
        self.root.mainloop()


def filter_code(settings):
    return (f"viewer_filter({settings['min_distance']}, {settings['min_angle']}, "
            f"{int(settings['min_frames'])})")


def connect(settings):
    """Open the kernel and push the remote half of the viewer into it."""
    kernel = KernelClient(settings["url"], settings["token"],
                          settings["kernel_id"]).connect()
    print(f"kernel {kernel.kernel_id}")

    # chdir so the kernel resolves "exp_navsim/config.yaml" like the shell does,
    # and put the repo on sys.path in case the kernel was started elsewhere.
    repo = settings["repo"]
    kernel.execute(f"import os, sys\n"
                   f"os.chdir({repo!r})\n"
                   f"sys.path.insert(0, {repo!r}) if {repo!r} not in sys.path else None\n")
    kernel.execute((Path(__file__).parent / "remote_episode.py").read_text())
    options = {k: settings[k] for k in _REMOTE_OPTS}
    print(kernel.call_json(f"viewer_config(**{options!r})"))
    return kernel


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Every flag defaults to its SETTINGS entry, so the dict stays the one place
    # to change a default.
    for name, value in SETTINGS.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            ap.add_argument(flag, dest=name, action="store_true", default=value)
            ap.add_argument("--no-" + name.replace("_", "-"), dest=name,
                            action="store_false")
        else:
            kind = type(value) if value is not None else str
            ap.add_argument(flag, dest=name, type=kind, default=value)
    SETTINGS.update(vars(ap.parse_args()))

    kernel = connect(SETTINGS)
    prefetcher = Prefetcher(kernel, SETTINGS)
    prefetcher.start()
    # Both are queued, so the window opens immediately and the slow work (loading
    # the checkpoint, scanning the dataset) reports itself in the status line.
    if SETTINGS["ckpt"]:
        prefetcher.commands.put(("model", f"viewer_model({SETTINGS['ckpt']!r}, "
                                          f"{SETTINGS['config']!r}, "
                                          f"{SETTINGS['pred_split']!r})"))
    prefetcher.commands.put(("filter", filter_code(SETTINGS)))
    Viewer(prefetcher, SETTINGS).run()
    kernel.close()


if __name__ == "__main__":
    main()
