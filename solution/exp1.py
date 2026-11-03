"""실험 1: 기준선 재현 vs 전체 특징 L2 vs L1 (참조 구성, 검증 2024-06~2025-05)."""
import numpy as np
import lightgbm as lgb
from common import VAL_FROM, load_train, train_smp, score_groups, group_frame, ref_groups, log

d = load_train()
smp = train_smp()
grp = ref_groups()
tr = d[(d.ts < VAL_FROM) & (d.elev > -3)]
va = d[d.ts >= VAL_FROM]
log(f"train {len(tr):,} / val {len(va):,}")

EXCL = {"site_id", "ts", "energy", "cf", "current_group", "wa_station", "wb_station"}
ALL = [c for c in d.columns if c not in EXCL]
BASE_WA = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_wind_speed", "wa_ground_press"]
BASE_WB = ["wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob",
           "wb_total_liq", "wb_vis", "wb_wind_speed", "wb_dew_point"]
BASE_OTHER = ["elev", "azim", "cosz", "doy_sin", "doy_cos", "hour", "capacity_kw", "lat", "lon"]


def fit(X, y, obj, n=800, lr=0.05):
    return lgb.LGBMRegressor(objective=obj, n_estimators=n, learning_rate=lr, num_leaves=127,
                             min_child_samples=200, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.7, verbose=-1, n_jobs=8).fit(X, y)


def report(name, cf_pred):
    kwh = np.clip(cf_pred, 0, None) * va.capacity_kw.to_numpy()
    kwh = np.where(va.elev.to_numpy() > -3, kwh, 0)
    r = score_groups(group_frame(va, kwh, grp), smp)
    log(f"{name:28s} penalty {r['pen']/1e6:8.3f}M원  ({r['pen_pct']:.3f}% of revenue)")
    return kwh


# 0) 기준선 재현: 라운드별 모델 L2 → 평균
preds = []
for r in (10, 17):
    cols = [f"{c}_r{r}" for c in BASE_WA + BASE_WB] + BASE_OTHER
    m = fit(tr[cols], tr.cf, "regression", n=600, lr=0.06)
    preds.append(m.predict(va[cols]))
report("baseline(round avg, L2)", np.mean(preds, axis=0))

# 1) 전체 특징 L2 / L1
for obj in ("regression", "l1"):
    m = fit(tr[ALL], tr.cf, obj)
    kwh = report(f"all feats {obj}", m.predict(va[ALL]))
    np.save(f"../cache/val_{obj}.npy", kwh)

imp = sorted(zip(m.booster_.feature_importance("gain"), ALL), reverse=True)[:25]
print([f"{n}:{g/1e3:.0f}" for g, n in imp])
