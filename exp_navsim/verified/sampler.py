"""One sampling procedure per spec string, shared by the 18_verified_sampling scripts.

    run/checkpoint[@mode][@tau][@g<w>]

    mode     prop (default), strat or nucleus<p> (IntentFlow.choose_anchors)
    tau      rejection threshold of the verifier on the predicted EPDMS: draw 4 x K
             futures, keep the first K that pass (diverse.verifier.filtered_sample);
             "-" or missing = no rejection
    g<w>     verifier guidance with weight w during the Euler steps (guided_sample)

e.g. 17_full_h64_full/step0100000@nucleus0.8@0.5, 17_full_h64_full/step0100000@prop@-@g3.

Guidance is classifier guidance for flow matching: at every Euler step the clean future
is estimated from the current velocity (x0 = z_t + t v), decoded, scored by the verifier,
and z moves along the gradient of log(predicted EPDMS) w.r.t. z_t, scaled by w and the
step size. It pushes every sample towards acceptable futures instead of throwing away
the ones that are not (Dhariwal & Nichol 2021; for flows the gradient guidance family of
"On the Guidance of Flow Matching", arXiv 2502.02150).
"""

from pathlib import Path

import torch

from exp_navsim.diverse.plot_bev import load_model
from exp_navsim.diverse.train import batch_of
from exp_navsim.diverse.verifier import epdms_torch, filtered_sample, load_verifier

VERIFIER = "logs_exp/17_verifier_gen/verifier.pt"


def parse(spec):
    parts = spec.split("@")
    out = {"stem": parts[0], "mode": "prop", "tau": None, "guide": 0.0}
    for p in parts[1:]:
        if p.startswith("g"):
            out["guide"] = float(p[1:])
        elif p in ("prop", "strat") or p.startswith("nucleus"):
            out["mode"] = p
        elif p != "-":
            out["tau"] = float(p)
    return out


def guided_sample(model, verifier, batch, num_samples, steps, mode, weight, generator=None):
    """IntentFlow / CtrlFlow sampling with verifier guidance -> futures (K, B, 8, 3)."""
    with torch.no_grad():
        base, kin, logits = model.encode_scene(batch)
        ks = model.choose_anchors(logits, num_samples, mode, generator=generator)
        K, B = ks.shape
        k = ks.flatten()
        kin_k = kin.repeat(K, 1) if kin is not None else None
        cond = model.flow_condition(base.repeat(K, 1, 1), kin_k, k)
        ctx = verifier.scene.condition({"features": batch["features"]}).repeat(K, 1, 1)
    z = torch.randn(K * B, model.codec.latent_dim, device=base.device, generator=generator)
    ts = torch.linspace(1, 0, steps + 1, device=base.device)
    for i in range(steps):
        dt = ts[i] - ts[i + 1]
        with torch.no_grad():
            v = model.denoiser(cond, z, ts[i].expand(K * B))
        if weight > 0:
            zg = z.detach().requires_grad_(True)
            future = model.decode(model.codec.decode(zg + ts[i] * v), kin_k, k)
            q = verifier.traj((future.flatten(1) - verifier.traj_mean) / verifier.traj_std)[:, None]
            for blk in verifier.blocks:
                q = blk(ctx, q)
            score = epdms_torch(torch.sigmoid(verifier.head(q)[:, 0]))
            grad, = torch.autograd.grad(torch.log(score + 1e-3).sum(), zg)
            v = v + weight * grad
        z = (z + v * dt).detach()
    with torch.no_grad():
        future = model.decode(model.codec.decode(z), kin_k, k)
    return future.unflatten(0, (K, B))


class Sampler:
    def __init__(self, spec, verifier=VERIFIER, steps=20, oversample=4):
        self.spec = parse(spec)
        run, ck = self.spec["stem"].split("/")
        self.model, self.args = load_model(Path("logs_exp") / run / "checkpoints" / f"{ck}.pt")
        need = self.spec["tau"] is not None or self.spec["guide"] > 0
        self.verifier = load_verifier(verifier) if need else None
        self.steps, self.oversample = steps, oversample

    def _one(self, b, K, g):
        s = self.spec
        if s["guide"] > 0:
            n = K * (self.oversample if s["tau"] is not None else 1)
            fut = guided_sample(self.model, self.verifier, b, n, self.steps, s["mode"], s["guide"], g)
            if s["tau"] is None:
                return fut
            with torch.no_grad():
                pred = self.verifier.predict_epdms({"features": b["features"]}, fut)
                order = torch.arange(len(pred), device=pred.device, dtype=pred.dtype)[:, None]
                top = torch.where(pred >= s["tau"], 2 - order / len(pred), pred).topk(K, dim=0).indices
            return fut.gather(0, top[..., None, None].expand(-1, -1, *fut.shape[2:]))
        with torch.no_grad():
            return filtered_sample(self.model, self.verifier, b, K, self.steps, s["mode"],
                                   self.oversample if s["tau"] is not None else 1, g, s["tau"])

    def __call__(self, data, K=32, seed=0, chunk=125):
        """data: to_gpu dict -> futures (K, N, 8, 3) on the GPU."""
        g = torch.Generator(device="cuda").manual_seed(seed)
        n = len(data["row"])
        return torch.cat([self._one(batch_of(data, torch.arange(i, min(i + chunk, n), device="cuda")), K, g)
                          for i in range(0, n, chunk)], 1)
