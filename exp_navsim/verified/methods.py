"""The methods of the 18_verified_sampling report: one key, one display name, one sampler spec.

Every table and figure of report_pdf.py uses these names. All generators are trained on
all 92k navtrain scenes for 100k steps (17_diversity), image features only (wm11_ctx);
the verifier is logs_exp/18_verifier_v3.
"""

S = "step0100000"
VERIFIER_V3 = "logs_exp/18_verifier_v3/verifier.pt"

# key, display name, sampler spec (sampler.py), short description
METHODS = [
    ("plain", "A  Plain flow", f"17_plain_h64_full/{S}",
     "flow-matching probe without anchors (the model that collapses on small data)"),
    ("intent", "B  Intent flow", f"17_full_h64_full/{S}",
     "speed head + 16 intent anchors + anchor-conditioned flow, anchor k ~ p(k | scene)"),
    ("nucleus", "C  B + nucleus", f"17_full_h64_full/{S}@nucleus0.8",
     "anchor drawn only from the most likely anchors with 80% of the mass"),
    ("guidance", "D  C + guidance", f"17_full_h64_full/{S}@nucleus0.8@-@g0.3",
     "verifier gradient added at every Euler step, weight 0.3"),
    ("rejection", "E  C + rejection", f"17_full_h64_full/{S}@nucleus0.8@0.5",
     "draw 128, keep the first 32 with predicted EPDMS >= 0.5"),
    ("all", "F  C + rejection + guidance", f"17_full_h64_full/{S}@nucleus0.8@0.5@g0.3",
     "both verifier steps"),
]
PLOTTED = tuple(m[0] for m in METHODS)                            # columns of the BEV pages


def by_key(key):
    return next(m for m in METHODS if m[0] == key)
