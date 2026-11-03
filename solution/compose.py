"""구성 최적화: 발전소별 잔차 e_s(t)로 Σ_t w_t(|E_G1|+|E_G2|) 최소화 (용량 제약 하 국소탐색).

검증: 2024-06~ 잔차를 홀수 주로 최적화 → 짝수 주에서 평가 (과적합 점검).
사용: python compose.py <val_kwh.npy>   (exp2가 저장한 검증 예측)
"""
import sys
import numpy as np
import pandas as pd
from common import VAL_FROM, load_train, train_smp, ref_groups, log

CAP_MIN = 20001.0   # 부동소수 안전여유 1 kW


def residual_matrix(va: pd.DataFrame, kwh: np.ndarray, smp: pd.DataFrame):
    t = va[["site_id", "ts", "energy"]].assign(e=va.energy.to_numpy() - kwh)
    E = t.pivot_table(index="ts", columns="site_id", values="e", aggfunc="sum").fillna(0.0)
    w = smp.set_index("ts").reindex(E.index)
    return E, (w.smp * w.k).fillna(0).to_numpy()


def objective(E: np.ndarray, w: np.ndarray, x: np.ndarray) -> float:
    e1 = E @ x
    e2 = E.sum(1) - e1
    return float(w @ (np.abs(e1) + np.abs(e2)))


def optimize(E: np.ndarray, w: np.ndarray, cap: np.ndarray, x0: np.ndarray, iters=2000):
    """단일 이동 + 교환 탐욕 국소탐색. x[s]=1 → G1."""
    x = x0.astype(float).copy()
    tot = E.sum(1)
    total_cap = cap.sum()
    for it in range(iters):
        e1 = E @ x
        cur = w @ (np.abs(e1) + np.abs(tot - e1))
        # 단일 이동: s를 반대편으로
        d = np.where(x > 0, -1.0, 1.0)            # e1 변화 부호
        ne1 = e1[:, None] + E * d[None, :]
        val = w @ (np.abs(ne1) + np.abs(tot[:, None] - ne1))
        c1 = cap @ x + cap * d
        ok = (c1 >= CAP_MIN) & (total_cap - c1 >= CAP_MIN)
        val = np.where(ok, val, np.inf)
        best = int(np.argmin(val))
        if val[best] < cur - 1e-6:
            x[best] = 1 - x[best]
            continue
        # 교환: G1의 i ↔ G2의 j (상위 후보만)
        g1, g2 = np.where(x > 0)[0], np.where(x == 0)[0]
        improved = False
        for i in g1[np.argsort(val[g1])[:30]]:
            ne1 = e1[:, None] - E[:, [i]] + E[:, g2]
            v = w @ (np.abs(ne1) + np.abs(tot[:, None] - ne1))
            c1 = cap @ x - cap[i] + cap[g2]
            v = np.where((c1 >= CAP_MIN) & (total_cap - c1 >= CAP_MIN), v, np.inf)
            j = int(np.argmin(v))
            if v[j] < cur - 1e-6:
                x[i], x[g2[j]] = 0, 1
                improved = True
                break
        if not improved:
            break
    return x


if __name__ == "__main__":
    d = load_train()
    va = d[d.ts >= VAL_FROM].reset_index(drop=True)
    kwh = np.load(sys.argv[1])
    smp = train_smp()
    E, w = residual_matrix(va, kwh, smp)
    sites = E.columns
    cap = d.groupby("site_id").capacity_kw.first().reindex(sites).to_numpy()
    week = E.index.isocalendar().week.to_numpy()
    odd, even = week % 2 == 1, week % 2 == 0
    M = E.to_numpy()
    ref = (ref_groups().reindex(sites) == "G1").to_numpy().astype(float)
    pen = lambda x, m: 0.10 * objective(M[m], w[m], x) / 1e6
    lb = lambda m: 0.10 * float(w[m] @ np.abs(M[m].sum(1))) / 1e6
    log(f"reference: odd {pen(ref, odd):.3f}M even {pen(ref, even):.3f}M | 하한 odd {lb(odd):.3f} even {lb(even):.3f}")
    for name, x0 in [("ref start", ref)]:
        x = optimize(M[odd], w[odd], cap, x0)
        log(f"opt(odd)  : odd {pen(x, odd):.3f}M even {pen(x, even):.3f}M  G1 {cap@x:.0f}kW, 이동 {int((x!=x0).sum())}")
    x = optimize(M, w, cap, ref)
    log(f"opt(full) : full {pen(x, odd|even):.3f}M (ref {pen(ref, odd|even):.3f}M)")
