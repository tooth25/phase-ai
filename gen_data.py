"""CALPHAD(COST507) 평형 계산으로 상 예측 학습 데이터셋 생성.

출력: data/<SYSTEM>.parquet  (원소 몰분율, T, 안정상 목록, 상분율)
"""
import itertools, os, sys, warnings, json
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

TDB = "COST507.tdb"
BINARIES = ["AL-CU", "AL-SI", "CU-SI", "CU-ZN", "CU-NI", "CU-MG", "AL-MG", "AL-ZN",
            "MG-ZN", "MG-SI", "SI-ZN", "CU-FE", "AL-FE", "FE-SI", "AL-MN", "MN-SI",
            "CU-ZR", "AL-TI", "SI-TI", "AL-LI", "CU-LI", "MG-NI", "AL-SN", "SN-ZN",
            "SI-SN", "CR-CU", "CU-Y", "AL-ZR", "SI-ZR", "MG-Y", "AL-CR"]
TERNARIES = ["AL-CU-SI", "AL-CU-MG", "AL-MG-SI", "AL-MG-ZN", "AL-CU-ZN",
             "AL-FE-SI", "AL-MN-SI", "CU-MG-ZN"]
T_BIN = np.arange(300, 1801, 20.0)
T_TER = np.arange(300, 1501, 50.0)


def base(name):
    return name.split("#")[0]


def run(system):
    from pycalphad import Database, equilibrium, variables as v
    out = f"data/{system}.parquet"
    if os.path.exists(out):
        return system, "skip"
    db = Database(TDB)
    els = system.split("-")
    comps = els + ["VA"]
    rows = []
    if len(els) == 2:
        conds = {v.X(els[1]): (1e-4, 1.0, 0.01), v.T: (T_BIN[0], T_BIN[-1] + 1, 20),
                 v.P: 101325, v.N: 1}
        eq = equilibrium(db, comps, list(db.phases), conds)
        xs = eq.coords[f"X_{els[1]}"].values
        Ts = eq.coords["T"].values
        PH = eq.Phase.values.squeeze()   # (T, X, vertex)
        NP = eq.NP.values.squeeze()
        for i, T in enumerate(Ts):
            for j, x in enumerate(xs):
                fr = {}
                for p, n in zip(PH[i, j], NP[i, j]):
                    if p and np.isfinite(n) and n > 1e-4:
                        fr[base(p)] = fr.get(base(p), 0) + float(n)
                if not fr:
                    continue
                rows.append({els[0]: 1 - x, els[1]: x, "T": T, "phases": json.dumps(fr)})
    else:
        step = 0.025
        a, b, c = els
        for xb in np.arange(0, 1 + 1e-9, step):
            # xb fixed, scan xc; ternary grid via successive 2-D calls
            xcs = np.arange(1e-4, 1 - xb - 1e-4 + 1e-9, step)
            if len(xcs) == 0:
                continue
            xbv = max(xb, 1e-4)
            try:
                eq = equilibrium(db, comps, list(db.phases),
                                 {v.X(b): xbv, v.X(c): (1e-4, max(1 - xbv - 1e-4, 2e-4), step),
                                  v.T: (T_TER[0], T_TER[-1] + 1, 50), v.P: 101325, v.N: 1})
            except Exception as e:
                continue
            xcv = eq.coords[f"X_{c}"].values
            Ts = eq.coords["T"].values
            PH = eq.Phase.values.squeeze(); NP = eq.NP.values.squeeze()
            if PH.ndim == 2:
                PH = PH[:, None, :]; NP = NP[:, None, :]
            for i, T in enumerate(Ts):
                for j, xc in enumerate(xcv):
                    if xbv + xc > 1:
                        continue
                    fr = {}
                    for p, n in zip(PH[i, j], NP[i, j]):
                        if p and np.isfinite(n) and n > 1e-4:
                            fr[base(p)] = fr.get(base(p), 0) + float(n)
                    if not fr:
                        continue
                    rows.append({a: 1 - xbv - xc, b: xbv, c: xc, "T": T, "phases": json.dumps(fr)})
    df = pd.DataFrame(rows)
    df["system"] = system
    df.to_parquet(out)
    return system, len(df)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    jobs = TERNARIES + BINARIES     # 느린 삼원계를 먼저 투입
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()) as pool:
        for s, n in pool.imap_unordered(run, jobs):
            print(s, n, flush=True)
