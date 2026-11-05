"""조성·온도 → 특성 벡터.

원소 몰분율(14원소) + T + 조성가중 원소 기술자(평균·표준편차·차이).
원소 기술자는 문헌 표준값의 근사치(금속 반경 pm, Pauling 전기음성도, VEC, 융점 K).
"""
import numpy as np

ELEM = {  # radius, EN, VEC, Tm
    "AL": (143, 1.61, 3, 933), "CU": (128, 1.90, 11, 1358), "SI": (117, 1.90, 4, 1687),
    "ZN": (134, 1.65, 12, 693), "NI": (124, 1.91, 10, 1728), "MG": (160, 1.31, 2, 923),
    "FE": (126, 1.83, 8, 1811), "MN": (127, 1.55, 7, 1519), "ZR": (160, 1.33, 4, 2128),
    "TI": (147, 1.54, 4, 1941), "LI": (152, 0.98, 1, 454), "SN": (158, 1.96, 4, 505),
    "CR": (128, 1.66, 6, 2180), "Y": (180, 1.22, 3, 1799),
    # 학습 데이터 없음(COST507 미수록): 입력은 가능하나 예측 신뢰도 낮음
    "GA": (135, 1.81, 3, 303), "GE": (125, 2.01, 4, 1211),
}
ELEMENTS = list(ELEM)
D = np.array([ELEM[e] for e in ELEMENTS], dtype=float)  # (n_el, 4)


def featurize(X, T):
    """X: (n, n_el) 몰분율, T: (n,) K"""
    X = np.asarray(X, float); T = np.asarray(T, float)
    mean = X @ D                                  # 조성 평균
    var = X @ D**2 - mean**2
    std = np.sqrt(np.clip(var, 0, None))
    delta = std[:, 0] / mean[:, 0]                # 원자크기 불일치 δ
    Tm = mean[:, 3]
    feats = [X, T[:, None] / 1000, (T / Tm)[:, None], mean / [100, 1, 10, 1000],
             std / [10, 1, 10, 1000], delta[:, None],
             (X > 1e-3).sum(1, keepdims=True)]
    return np.hstack(feats)
