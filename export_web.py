"""웹 데모용: MLP 가중치·스케일러·CALPHAD 기준 격자를 JSON으로 내보냄."""
import json, glob, base64, numpy as np, pandas as pd, joblib
from features import ELEMENTS, D
b = joblib.load("model.joblib"); pipe = b["model"]
sc, mlp = pipe.steps[0][1], pipe.steps[1][1]
enc = lambda a: base64.b64encode(np.asarray(a, np.float32).tobytes()).decode()
layers = [{"W": enc(W), "b": enc(c), "shape": list(W.shape)} for W, c in zip(mlp.coefs_, mlp.intercepts_)]
ref = {}
for f in sorted(glob.glob("data/*.parquet")):
    df = pd.read_parquet(f); s = df.system.iloc[0]; els = s.split("-")
    if len(els) != 2: continue
    df = df[(np.round(df[els[1]] * 100).astype(int) % 2 == 0) & ((df["T"] - 300) % 40 == 0)]
    ref[s] = [[round(float(r[els[1]]), 3), float(r["T"]), sorted(json.loads(r.phases))] for _, r in df.iterrows()]
rep = json.load(open("report.json"))
out = {"elements": ELEMENTS, "desc": D.tolist(), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
       "layers": layers, "classes": list(b["classes"]), "ref": ref, "report": rep}
json.dump(out, open("web_model.json", "w"), separators=(",", ":"))
print(len(json.dumps(out)) / 1e6, "MB", mlp.activation, mlp.out_activation_)
