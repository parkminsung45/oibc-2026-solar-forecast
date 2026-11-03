"""최종 제출 생성.

1) 구성용 잔차: 2024-06~2025-05를 주(week mod 4)로 4분할, 나머지 전 기간으로 학습해 OOF 잔차
2) 잔차로 구성 최적화 (참조 구성에서 출발하는 국소탐색, 각 집합 ≥ 20,000 kW)
3) 전 기간 학습 앙상블로 평가 구간 발전소별 예측 → 집합 합계
출력: solution/submission/{composition,forecast}.csv, cache/test_site_pred.parquet
"""
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import TZ, VAL_FROM, load_train, train_smp, ref_groups, log
from features import DATA, get
from compose import residual_matrix, optimize, objective

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission")
os.makedirs(OUT, exist_ok=True)
MODELS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["lgb0", "lgb1", "lgb2"]
TRAIN_END = pd.Timestamp("2025-06-01 01:00", tz=TZ)
EXCL = {"site_id", "ts", "energy", "cf", "current_group", "wa_station", "wb_station"}


def target(df, ref_time):
    den = np.maximum(df.csky.to_numpy(), 0.02)
    y = np.clip(df.cf.to_numpy() / den, 0, 2)
    age = (ref_time - df.ts).dt.days.to_numpy()
    return y, np.exp(-age / 730) * den ** 2


def make(name):
    if name.startswith("lgb"):
        return lgb.LGBMRegressor(objective="regression", n_estimators=800, learning_rate=0.05,
                                 num_leaves=127, min_child_samples=200, subsample=0.8,
                                 subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=8,
                                 random_state=int(name[3:] or 0))
    if name == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=9,
                                min_child_weight=50, subsample=0.8, colsample_bytree=0.7,
                                tree_method="hist", n_jobs=8)
    if name == "cat":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=2000, learning_rate=0.08, depth=8, verbose=0,
                                 thread_count=8)
    raise ValueError(name)


def predict_kwh(models, X: pd.DataFrame, feats):
    r = np.mean([m.predict(X[feats]) for m in models], axis=0)
    kwh = np.clip(r * np.maximum(X.csky.to_numpy(), 0.02), 0, None) * X.capacity_kw.to_numpy()
    return np.where(X.elev.to_numpy() > -3, kwh, 0.0)


d = load_train()
FEATS = [c for c in d.columns if c not in EXCL]
smp = train_smp()
day = d.elev > -3

# 1) 구성용 OOF 잔차 (검증 연도, 주 단위 4분할, lgb 1개)
if os.path.exists("../cache/compose_oof.npy"):
    oof = np.load("../cache/compose_oof.npy")
else:
    va_mask = d.ts >= VAL_FROM
    fold = d.ts.dt.isocalendar().week.to_numpy() % 4
    oof = np.zeros(len(d))
    for f in range(4):
        te = va_mask & (fold == f)
        trm = day & ~te
        y, w = target(d[trm], TRAIN_END)
        m = make("lgb0").fit(d.loc[trm, FEATS], y, sample_weight=w)
        oof[te.to_numpy()] = predict_kwh([m], d[te], FEATS)
        log(f"compose fold {f} done")
    np.save("../cache/compose_oof.npy", oof)

# 2) 구성 최적화
va = d[d.ts >= VAL_FROM].reset_index(drop=True)
E, wt = residual_matrix(va, oof[(d.ts >= VAL_FROM).to_numpy()], smp)
sites = E.columns
cap = d.groupby("site_id").capacity_kw.first().reindex(sites).to_numpy()
ref = (ref_groups().reindex(sites) == "G1").to_numpy().astype(float)
M = E.to_numpy()
x = optimize(M, wt, cap, ref)
log(f"구성 페널티(검증 OOF): ref {0.1*objective(M, wt, ref)/1e6:.3f}M → opt {0.1*objective(M, wt, x)/1e6:.3f}M"
    f" | 하한 {0.1*float(wt @ np.abs(M.sum(1)))/1e6:.3f}M | G1 {cap@x:.1f} kW, G2 {cap.sum()-cap@x:.1f} kW")
groups = pd.Series(np.where(x > 0, "G1", "G2"), index=sites)

# 3) 전 기간 학습 → 평가 구간 예측
y, w = target(d[day], TRAIN_END)
Xtr = d.loc[day, FEATS]
del oof
te = get("test")
preds = []
for name in MODELS:
    m = make(name).fit(Xtr, y, sample_weight=w)
    p = predict_kwh([m], te, FEATS)
    preds.append(p)
    np.save(f"../cache/test_pred_{name}.npy", p)
    log(f"model {name} trained, test total {p.sum()/1e6:.2f} GWh")
te["kwh"] = np.mean(preds, axis=0)
te[["site_id", "ts", "kwh"]].to_parquet("../cache/test_site_pred.parquet", index=False)

# 4) 제출 파일
index = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(f"{DATA}/smp.csv").ts)).tz_convert(TZ)
te["group"] = te.site_id.map(groups)
agg = te.groupby(["group", "ts"]).kwh.sum()
rows = []
for g in ("G1", "G2"):
    s = agg.get(g, pd.Series(dtype=float)).reindex(index, fill_value=0.0).clip(lower=0)
    rows.append(pd.DataFrame({"group": g, "ts": index.strftime("%Y-%m-%d %H:%M:%S%z")
                              .str.replace(r"(\d{2})(\d{2})$", r"\1:\2", regex=True),
                              "forecast_kwh": s.round(3).to_numpy()}))
fc = pd.concat(rows, ignore_index=True)
fc.to_csv(f"{OUT}/forecast.csv", index=False)
groups.rename("group").rename_axis("site_id").reset_index().sort_values("site_id") \
      .to_csv(f"{OUT}/composition.csv", index=False)
log(f"저장: {OUT}/composition.csv, forecast.csv ({len(fc)}행)")
