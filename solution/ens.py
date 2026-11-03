"""cv 예측 앙상블 가중치 탐색 (A+B 합 페널티 최소, 비음수·합 1).

사용: python ens.py name1 name2 ...
"""
import sys
import itertools
import numpy as np
from tune import FOLDS, CV, score, log

names = sys.argv[1:]
P = {f: np.stack([np.load(f"{CV}/{n}_{f}.npy") for n in names]) for f in FOLDS}


def total(w):
    return sum(score(f, w @ P[f]) for f in FOLDS)


eq = np.full(len(names), 1 / len(names))
log(f"균등 {total(eq):.3f}M")
# 좌표 하강 (0.05 단위)
w = eq.copy()
best = total(w)
step = 0.1
while step >= 0.025:
    improved = False
    for i, j in itertools.permutations(range(len(names)), 2):
        if w[j] < step - 1e-9:
            continue
        v = w.copy()
        v[i] += step
        v[j] -= step
        s = total(v)
        if s < best - 1e-4:
            w, best, improved = v, s, True
    if not improved:
        step /= 2
log("최적 " + ", ".join(f"{n}={x:.3f}" for n, x in zip(names, w)) + f" → {best:.3f}M "
    f"(A {score('A', w @ P['A']):.3f}, B {score('B', w @ P['B']):.3f})")
