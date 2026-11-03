"""최종 제출 v2: GBDT 5 + NN 8 앙상블(40:60), 2년치 OOF 앙상블 오차로 구성 최적화.

1) 구성: cache/oof2/{lgb,xgb,nn_e4_bs16k,nn_e4_bs16k_p3}.npy (oof2.py) → 앙상블 잔차
   교차 연도 홀드아웃으로 점검한 뒤 2년 전체로 최적화
2) 전 기간(~2025-05) 학습 → 평가 구간 발전소별 예측 (testpred.py, cache/test2/ 재사용)
3) submission/{composition,forecast}.csv
"""
import os
import numpy as np
import pandas as pd
from common import TZ, ref_groups, log
from features import DATA
from tune import data
from testpred import predict, test_frame
from oof2 import OUT as OOF, span_mask
from compose import residual_matrix, optimize, objective

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = f"{ROOT}/solution/submission"
os.makedirs(SUB, exist_ok=True)
GBDT = ["lgb", "lgb_s1", "lgb_lr03", "xgb", "xgb_lr03"]
NN = ["nn_e4_bs16k", "nn_e4_bs16k_s1", "nn_e4_bs16k_s2", "nn_e4_bs16k_p3", "nn_e3_bs16k",
      "nn_e4_bs16k_p3_s1", "nn_e4_bs16k_p3_s2", "nn_e3_bs16k_s1"]


def blend(p):
    """p: name → 예측. GBDT 평균 40% + NN 평균 60% (2폴드 검증, 기압 제외 기준)."""
    g = np.mean([p[n] for n in GBDT if n in p], 0)
    n = np.mean([p[n] for n in NN if n in p], 0)
    return 0.4 * g + 0.6 * n


D = data()
d, smp = D["d"], D["smp"]

# 1) 구성
span = span_mask(d)
oof = blend({n: np.load(f"{OOF}/{n}.npy")[span] for n in ["lgb", "xgb", "nn_e4_bs16k", "nn_e4_bs16k_p3"]})
va = d[span].reset_index(drop=True)
E, wt = residual_matrix(va, oof, smp)
sites = E.columns
cap = d.groupby("site_id").capacity_kw.first().reindex(sites).to_numpy()
ref = (ref_groups().reindex(sites) == "G1").to_numpy().astype(float)
M = E.to_numpy()
pen = lambda x, m: 0.1 * objective(M[m], wt[m], x) / 1e6
y1 = (E.index < pd.Timestamp("2024-06-01 01:00", tz=TZ))
h = E.index.month.isin([6, 7, 8, 9, 10, 11])
for nm, a, b in [("연1→연2", y1, ~y1), ("연2→연1", ~y1, y1),
                 ("6~11월→12~5월", h, ~h), ("12~5월→6~11월", ~h, h)]:
    x = optimize(M[a], wt[a], cap, ref)
    log(f"구성 홀드아웃 {nm}: ref {pen(ref, b):.3f}M → opt {pen(x, b):.3f}M")
x = optimize(M, wt, cap, ref)
log(f"구성(2년 OOF): ref {pen(ref, slice(None)):.3f}M → opt {pen(x, slice(None)):.3f}M | "
    f"G1 {cap@x:.1f} kW ({int(x.sum())}개), G2 {cap.sum()-cap@x:.1f} kW, 이동 {int((x != ref).sum())}개")
groups = pd.Series(np.where(x > 0, "G1", "G2"), index=sites)

# 2) 전 기간 학습 → 평가 구간 예측
te = test_frame()
preds = {name: predict(f"{name}_np") for name in GBDT + NN}   # 기압 제외 (tune.PRESS)
te["kwh"] = blend(preds)
te[["site_id", "ts", "kwh"]].to_parquet(f"{ROOT}/cache/test_site_pred2.parquet", index=False)

# 3) 제출 파일
index = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(f"{DATA}/smp.csv").ts)).tz_convert(TZ)
te["group"] = te.site_id.map(groups)
assert te.group.notna().all()
agg = te.groupby(["group", "ts"]).kwh.sum()
rows = []
for g in ("G1", "G2"):
    s = agg.get(g, pd.Series(dtype=float)).reindex(index, fill_value=0.0).clip(lower=0)
    rows.append(pd.DataFrame({"group": g, "ts": index.strftime("%Y-%m-%d %H:%M:%S%z")
                              .str.replace(r"(\d{2})(\d{2})$", r"\1:\2", regex=True),
                              "forecast_kwh": s.round(3).to_numpy()}))
fc = pd.concat(rows, ignore_index=True)
fc.to_csv(f"{SUB}/forecast.csv", index=False)
groups.rename("group").rename_axis("site_id").reset_index().sort_values("site_id") \
      .to_csv(f"{SUB}/composition.csv", index=False)
log(f"저장: {SUB} (forecast {len(fc)}행, 합계 {fc.forecast_kwh.sum()/1e6:.2f} GWh)")
