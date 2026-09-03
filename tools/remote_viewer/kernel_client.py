"""Minimal Jupyter kernel client — drives a remote kernel without a notebook.

A Jupyter kernel is just a process speaking a documented message protocol, so
anything can drive it. There are two ways in:

  * direct ZMQ (`jupyter_client.BlockingKernelClient` + the kernel's connection
    file) — needs all FIVE kernel ports SSH-forwarded and the connection file
    rewritten to point at 127.0.0.1;
  * through the Jupyter *server* you already run, which proxies the kernel over
    REST + a websocket at /api/kernels/<id>/channels — ONE port, reusing the
    tunnel and the token you already have.

This module takes the second route. Images come back as `display_data` messages
carrying base64 JPEG (the protocol is JSON, so binary must be base64 — ~33%
overhead, which is why the viewer prefetches rather than fetching per-frame).

Requires: pip install websocket-client
"""

import json
import urllib.request
import uuid


class RemoteError(RuntimeError):
    """The remote kernel raised — carries its traceback."""


class KernelClient:
    """One websocket to one kernel on a Jupyter server.

    NOT thread-safe: `execute` blocks until its own request goes idle, so all
    calls must come from a single thread (in the viewer, the prefetcher thread).
    """

    def __init__(self, base_url="http://localhost:8888", token="", kernel_id=None,
                 timeout=300):
        self.base = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.session = uuid.uuid4().hex          # our client id in the kernel's session
        self.kernel_id = kernel_id or self._pick_kernel()
        self.ws = None

    # --- REST side ----------------------------------------------------------- #
    def _api(self, path, payload=None):
        """Call /api/<path>; passing `payload` makes it a POST."""
        req = urllib.request.Request(
            f"{self.base}/api/{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Authorization": f"token {self.token}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def _pick_kernel(self):
        """Reuse the server's first running kernel, or start a fresh python3 one.

        Reusing matters: a kernel that already holds the dataset (and any loaded
        model) makes seeking instant, which is the whole reason to go through a
        kernel instead of shelling out a fresh process per frame.
        """
        running = self._api("kernels")
        return running[0]["id"] if running else self._api("kernels", {"name": "python3"})["id"]

    def list_kernels(self):
        return self._api("kernels")

    # --- websocket side ------------------------------------------------------ #
    def connect(self):
        import websocket                        # pip install websocket-client

        # http -> ws / https -> wss, only the scheme prefix.
        url = (f"{self.base.replace('http', 'ws', 1)}"
               f"/api/kernels/{self.kernel_id}/channels?session_id={self.session}")
        # Jupyter Server checks Origin on websocket upgrades, so send a matching one.
        self.ws = websocket.create_connection(
            url,
            header=[f"Authorization: token {self.token}"],
            origin=self.base,
            timeout=self.timeout,
        )
        return self

    def close(self):
        if self.ws is not None:
            self.ws.close()
            self.ws = None

    def execute(self, code):
        """Run `code`; return its iopub output messages (stream/display/result).

        Filters on parent msg_id, so output produced by *other* clients sharing
        this kernel is ignored. Always drains to the closing `status: idle`
        before raising, otherwise a failed call would desync the stream and every
        later call would read the wrong messages.
        """
        msg_id = uuid.uuid4().hex
        self.ws.send(json.dumps({
            "header": {"msg_id": msg_id, "username": "viewer", "session": self.session,
                       "msg_type": "execute_request", "version": "5.3"},
            "parent_header": {}, "metadata": {},
            "content": {"code": code, "silent": False, "store_history": False,
                        "user_expressions": {}, "allow_stdin": False,
                        "stop_on_error": True},
            "channel": "shell",
        }))

        outputs, error = [], None
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            kind = msg["header"]["msg_type"]
            if kind == "error":
                error = "\n".join(msg["content"]["traceback"])
            elif kind == "status" and msg["content"]["execution_state"] == "idle":
                break                                    # our request is complete
            elif kind in ("stream", "display_data", "execute_result"):
                outputs.append(msg)
        if error:
            raise RemoteError(error)
        return outputs

    # --- helpers used by the viewer ------------------------------------------ #
    def call_json(self, code):
        """Run `code`, which must print exactly one JSON object, and parse it."""
        text = "".join(m["content"]["text"] for m in self.execute(code)
                       if m["header"]["msg_type"] == "stream")
        for line in text.splitlines():
            if line.strip().startswith("{"):
                return json.loads(line)
        raise RemoteError(f"expected a JSON line, got:\n{text}")

    def fetch_frames(self, start, stop, quality, max_width, total_len, num_samples):
        """Render frames [start, stop) remotely.

        Returns ({index: JPEG bytes | error str}, note), where `note` is a
        non-fatal remote complaint (e.g. the model failed but the BEV still drew).

        One execute_request per *batch*: the SSH round trip (tens of ms) is
        amortised over `stop - start` frames instead of paid per frame, and the
        remote side gets to run the model for the whole span as one batch.
        """
        import base64

        msgs = self.execute(
            f"viewer_frames({start}, {stop}, quality={quality}, max_width={max_width}, "
            f"total_len={total_len}, num_samples={num_samples})")
        frames, note = {}, ""
        for m in msgs:
            # display_data puts metadata inside `content`, alongside `data`.
            tag = (m["content"].get("metadata") or {}).get("viewer")
            if not tag:
                continue
            if "note" in tag:               # went wrong, but cost no frame
                note = tag["note"]
            elif "error" in tag:
                frames[tag["index"]] = tag["error"]
            else:
                frames[tag["index"]] = base64.b64decode(m["content"]["data"]["image/jpeg"])
        return frames, note
