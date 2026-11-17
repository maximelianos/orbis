"""Shape and sanity checks of the v22 planner, on the CPU.

    python -m exp_navsim.v22.test_model
"""

import numpy as np
import torch

from exp_navsim.v22.model import (GaragePlanner, action_targets, path_targets, rollout, scaled,
                                  unscaled)


def fake_batch(B=5, T=1, C=64):
    g = torch.Generator().manual_seed(0)
    speed = torch.rand(B, generator=g) * 12
    # a constant-speed arc per scene, so the action and path targets have a known answer
    k = (torch.rand(B, generator=g) - 0.5) * 0.08
    t = torch.arange(1, 9) * 0.5
    th = k[:, None] * speed[:, None] * t
    x = (speed[:, None] * t) * torch.cos(th / 2)
    y = (speed[:, None] * t) * torch.sin(th / 2)
    return {"features": torch.randn(B, T, C, 9, 16, generator=g),
            "future": torch.stack([x, y, th], -1),
            "past": torch.randn(B, 4, 3, generator=g) * 2,
            "status": torch.stack([speed, torch.zeros(B), torch.zeros(B), torch.zeros(B)], -1),
            "command": torch.eye(4)[torch.randint(4, (B,), generator=g)]}


def main():
    b = fake_batch()
    print("scaled/unscaled round trip:", torch.allclose(unscaled(scaled(b["future"])), b["future"], atol=1e-5))

    p, mask = path_targets(b["future"])
    d = p.norm(dim=-1)
    print("path point distances (should be near 4, 8, ... where the mask is true):")
    for i in range(len(p)):
        print("  ", np.round(d[i].numpy(), 1), mask[i].numpy().astype(int))

    act = action_targets(b["future"], b["status"][:, 0])
    poses = rollout(act, b["status"][:, 0])
    err = (poses[..., :2] - b["future"][..., :2]).norm(dim=-1).mean(-1)
    print("rollout of the inverted actions, mean position error [m]:", np.round(err.numpy(), 3))

    for kw in (dict(groups=("wp",)), dict(groups=("wp", "act", "path")),
               dict(groups=("wp", "act", "path"), mask_path=False),
               dict(groups=("wp",), flow=False),
               dict(groups=("wp", "path"), vocab=np.random.randn(16, 8, 3).astype(np.float32),
                    score_head=True)):
        vocab = kw.get("vocab")
        m = GaragePlanner(64, 1, dim=32, depth=1, flow_depth=2, heads=4, registers=4, **kw)
        m.fit({"future": b["future"], "status": b["status"]})
        tgt = None
        if vocab is not None:
            tgt = (torch.rand(len(b["future"]), len(vocab), 8), torch.ones(len(b["future"]), dtype=torch.bool))
        loss, logs = m.loss(b, score_targets=tgt)
        loss.backward()
        grads = sum(int(p.grad is not None and p.grad.abs().sum() > 0) for p in m.parameters())
        s = m.sample(b, 3, steps=4, temperature=1.0)
        det = [m.sample(b, 1, steps=4, temperature=0.0) for _ in range(2)]
        plans = {k: m.predict(b, 4, plan=k, steps=4) for k in
                 (("flow", "vocab", "rerank") if vocab is not None else ("flow",))}
        print(f"{str(kw.get('groups'))} flow={kw.get('flow', True)} vocab={vocab is not None}: "
              f"loss {loss.item():.3f} {({k: round(v.item(), 3) for k, v in logs.items()})} "
              f"grads {grads}/{len(list(m.parameters()))} sample {tuple(s.shape)} "
              f"deterministic {torch.allclose(det[0], det[1])} "
              f"spread {(s.std(0).mean()).item():.3f} "
              + " ".join(f"{k}{tuple(v.shape)}" for k, v in plans.items()))


if __name__ == "__main__":
    main()
