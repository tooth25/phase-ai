"""상 예측 모델 학습·평가.

평가 두 가지:
  (1) 무작위 분할 — 이미 아는 계 내부의 보간 성능
  (2) 계 단위 제외(hold-out) — 학습에 없던 합금계에 대한 일반화 성능
"""
import glob, json, sys, time
import numpy as np, pandas as pd, joblib
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from features import ELEMENTS, featurize

MAX_TERNARY = None


def load():
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/*.parquet"))],
                   ignore_index=True)
    for e in ELEMENTS:
        if e not in df:
            df[e] = 0.0
    df[ELEMENTS] = df[ELEMENTS].fillna(0.0)
    df["phase_set"] = df.phases.map(lambda s: sorted(json.loads(s)))
    return df


def decode(proba, classes, n_comp, thr=0.5):
    """확률 → 상 집합. 깁스 상률(등압: 최대 C+1상)로 개수 제한."""
    out = []
    for p, c in zip(proba, n_comp):
        idx = np.argsort(-p)
        sel = [i for i in idx[: int(c) + 1] if p[i] >= thr] or [idx[0]]
        out.append(sorted(classes[i] for i in sel))
    return out


def predict_proba(model, F):
    return model.predict_proba(F)


def score(true_sets, pred_sets):
    exact = np.mean([a == b for a, b in zip(true_sets, pred_sets)])
    jac = np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(true_sets, pred_sets)])
    return exact, jac


def make_model(epochs=60):
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(256, 256, 128), batch_size=512, learning_rate_init=1e-3,
        max_iter=epochs, early_stopping=True, n_iter_no_change=8, random_state=0))


def main():
    df = load()
    print("samples", len(df), "systems", df.system.nunique())
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df.phase_set)
    classes = mlb.classes_
    print("phase labels", len(classes))
    F = featurize(df[ELEMENTS].values, df["T"].values)
    ncomp = (df[ELEMENTS].values > 1e-3).sum(1)
    report = {"n_samples": len(df), "n_systems": int(df.system.nunique()),
              "n_phases": len(classes)}

    import os
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    # (1) 무작위 분할
    rng = np.random.default_rng(0)
    te = rng.random(len(df)) < 0.2
    m = make_model().fit(F[~te], Y[~te])
    pred = decode(predict_proba(m, F[te]), classes, ncomp[te])
    ex, jc = score(df.phase_set[te].tolist(), pred)
    report["random_split"] = {"exact_match": ex, "jaccard": jc}
    print(f"[무작위 분할] 상 집합 완전일치 {ex:.3f}, Jaccard {jc:.3f}")

    # (2) 계 단위 hold-out: 삼원계를 하나씩 빼고, 해당 이원계로부터 외삽
    report["holdout"] = {}
    for sys_out in ["AL-CU-SI", "AL-CU-MG", "CU-MG-ZN"]:
        tr = df.system != sys_out
        m2 = make_model(25).fit(F[tr], Y[tr])
        te2 = ~tr
        pred2 = decode(predict_proba(m2, F[te2]), classes, ncomp[te2])
        ex2, jc2 = score(df.phase_set[te2].tolist(), pred2)
        report["holdout"][sys_out] = {"exact_match": ex2, "jaccard": jc2}
        print(f"[hold-out {sys_out}] 완전일치 {ex2:.3f}, Jaccard {jc2:.3f}", flush=True)

    # 최종 모델: 전체 데이터
    final = make_model().fit(F, Y)
    joblib.dump({"model": final, "classes": classes, "elements": ELEMENTS},
                "model.joblib", compress=3)
    sc, mlp = final.steps[0][1], final.steps[1][1]
    arr = {"mean": sc.mean_, "scale": sc.scale_, "classes": np.array([str(c) for c in classes], dtype="U32"),
           "elements": np.array(ELEMENTS, dtype="U4")}
    for i, (W, c) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        arr[f"W{i}"], arr[f"b{i}"] = W.astype(np.float32), c.astype(np.float32)
    np.savez_compressed("model.npz", **arr)
    json.dump(report, open("report.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
