"""조성·온도 → 예측 안정상.

사용 예
  python predict.py "Cu-5Al-2Si" 600          # at%, 첫 원소가 잔부, 온도 °C
  python predict.py "Al-4.5Cu" 500 --wt        # wt% 입력
  python predict.py "Cu-10Al" --map AL         # Cu-Al 이원 예측 상태도 그림(png)
"""
import argparse, re, sys
import numpy as np
from features import ELEMENTS, featurize

MASS = {"AL": 26.98, "CU": 63.55, "SI": 28.09, "ZN": 65.38, "NI": 58.69, "MG": 24.31,
        "FE": 55.85, "MN": 54.94, "ZR": 91.22, "TI": 47.87, "LI": 6.94, "SN": 118.71,
        "CR": 52.00, "Y": 88.91, "GA": 69.72, "GE": 72.63}
TRAINED = {"AL", "CU", "SI", "ZN", "NI", "MG", "FE", "MN", "ZR", "TI", "LI", "SN", "CR", "Y"}


def parse(s, wt=False):
    toks = re.findall(r"([\d.]*)([A-Z][a-z]?)", s)
    comp = {}
    for amt, el in toks[1:]:
        comp[el.upper()] = float(amt or 0)
    comp[toks[0][1].upper()] = 100 - sum(comp.values())
    if wt:
        mol = {e: w / MASS[e] for e, w in comp.items()}
        tot = sum(mol.values()); comp = {e: 100 * m / tot for e, m in mol.items()}
    return {e: v / 100 for e, v in comp.items()}


class _MLP:
    """numpy만으로 동작하는 추론기 (scikit-learn 버전과 무관)."""
    def __init__(self, z):
        self.mean, self.scale = z["mean"], z["scale"]
        n = sum(1 for k in z.files if k.startswith("W"))
        self.layers = [(z[f"W{i}"].astype(float), z[f"b{i}"].astype(float)) for i in range(n)]

    def predict_proba(self, F):
        h = (np.asarray(F, float) - self.mean) / self.scale
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            h = np.maximum(h, 0) if i < len(self.layers) - 1 else 1 / (1 + np.exp(-h))
        return h


def load(path=None):
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    z = np.load(path or os.path.join(here, "model.npz"), allow_pickle=False)
    return {"model": _MLP(z), "classes": [str(c) for c in z["classes"]]}


def predict(bundle, comp, T_K, thr=0.5):
    x = np.zeros((len(np.atleast_1d(T_K)), len(ELEMENTS)))
    for e, v in comp.items():
        x[:, ELEMENTS.index(e)] = v
    F = featurize(x, np.atleast_1d(T_K))
    return bundle["model"].predict_proba(F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alloy"); ap.add_argument("T_C", nargs="?", type=float, default=25)
    ap.add_argument("--wt", action="store_true"); ap.add_argument("--map")
    a = ap.parse_args()
    comp = parse(a.alloy, a.wt)
    unk = set(comp) - TRAINED
    if unk:
        print(f"[주의] {','.join(sorted(unk))}: 학습 DB에 없는 원소 — 원소 기술자만으로 외삽, 신뢰도 낮음")
    b = load()
    if a.map:
        return draw_map(b, list(comp)[-1] if a.map is True else a.map, comp)
    P = predict(b, comp, a.T_C + 273.15)[0]
    print("조성(at%):", ", ".join(f"{e} {100*v:.2f}" for e, v in comp.items()), f"| T = {a.T_C:.0f} °C")
    for i in np.argsort(-P)[:6]:
        if P[i] < 0.05: break
        print(f"  {b['classes'][i]:<14s} {P[i]:.2f}{'  ← 안정상' if P[i] >= 0.5 else ''}")


def draw_map(b, solute, comp):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    host = [e for e in comp if e != solute.upper()][0]
    xs = np.linspace(0, 1, 201); Ts = np.arange(300, 1801, 10.0)
    X, TT = np.meshgrid(xs, Ts)
    lab = {}
    grid = np.zeros(X.shape, int)
    for i, T in enumerate(Ts):
        x = np.zeros((len(xs), len(ELEMENTS)))
        x[:, ELEMENTS.index(host)] = 1 - xs; x[:, ELEMENTS.index(solute.upper())] = xs
        F = featurize(x, np.full(len(xs), T))
        P = b["model"].predict_proba(F)
        for j, p in enumerate(P):
            s = "+".join(sorted(b["classes"][k] for k in np.where(p >= .5)[0][:3])) or b["classes"][p.argmax()]
            grid[i, j] = lab.setdefault(s, len(lab))
    plt.figure(figsize=(9, 6))
    plt.pcolormesh(X * 100, TT - 273.15, grid, cmap="tab20", shading="auto")
    for s, k in lab.items():
        ii, jj = np.where(grid == k)
        if len(ii) > 40:
            plt.text(X[ii, jj].mean() * 100, TT[ii, jj].mean() - 273.15, s, fontsize=6, ha="center")
    plt.xlabel(f"at% {solute.upper()}"); plt.ylabel("T (°C)")
    plt.title(f"Predicted {host}-{solute.upper()} phase map (ML)")
    out = f"pred_{host}-{solute.upper()}.png"; plt.savefig(out, dpi=150, bbox_inches="tight")
    print("저장:", out)


if __name__ == "__main__":
    main()
