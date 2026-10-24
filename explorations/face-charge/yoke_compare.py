"""Magnet on plate: face-charge levels and MoM levels against the face-charge Richardson limit."""
import numpy as np, sys
from scipy.optimize import brentq
from yoke_case import sensors
obs, tags = sensors()

def richardson(levels, vals):
    (n1, b1), (n2, b2), (n3, b3) = list(zip(levels, vals))[-3:]
    d12, d23 = np.linalg.norm(b2 - b1), np.linalg.norm(b3 - b2)
    p = brentq(lambda p: (n1**-p - n2**-p) / (n2**-p - n3**-p) - d12 / d23, 0.2, 8)
    return b3 + (b3 - b2) * n3**-p / (n2**-p - n3**-p), p

if __name__ == "__main__":
    fc = np.load(sys.argv[1], allow_pickle=True).item()
    mr = {}
    for f in sys.argv[2:]:
        mr.update(np.load(f, allow_pickle=True).item())
    ms = sorted(fc); hs = sorted(mr, reverse=True)
    ref, p = richardson(ms, [fc[m] for m in ms])
    rel = lambda B: np.linalg.norm(B - ref, axis=1) / np.linalg.norm(ref, axis=1)
    names = sorted({t for t, _ in tags}, key=[t for t, _ in tags].index)
    print(f"reference: face-charge Richardson on m = {ms[-3:]}, observed order {p:.2f}; |ref| at sensors:",
          " ".join(f"{np.linalg.norm(ref[k])*1e3:.1f}" for k in range(len(tags))), "mT")
    print("errors per sensor (columns: " + ", ".join(names) + "; each at d = 0.5, 1, 2, 5 mm)")
    def row(label, B):
        e = rel(B)
        print(f"  {label:28s}", " | ".join(" ".join(f"{e[k]:.0e}" for k in range(j, len(tags), len(names))) for j in range(len(names))))
    for m in ms:
        row(f"face charge m={m}", fc[m])
    for h in hs:
        row(f"MoM h={h} mm", mr[h])
    if len(hs) >= 3:
        ext, q = richardson([1 / h for h in hs], [mr[h] for h in hs])
        row(f"MoM Richardson (order {q:.2f})", ext)
