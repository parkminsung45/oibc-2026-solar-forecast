"""실험 3: 가중치·타깃 변환·특징 제거·최근성 (주 지표: seen 발전소 페널티)."""
import sys
import numpy as np
import lightgbm as lgb
from common import VAL_FROM, load_train, train_smp, score_groups, group_frame, ref_groups, log

d = load_train()
smp = train_smp()
grp = ref_groups()
EXCL = {"site_id", "ts", "energy", "cf", "current_group", "wa_station", "wb_station"}
BASE = [c for c in d.columns if c not in EXCL]
tr = d[(d.ts < VAL_FROM) & (d.elev > -3)]
va = d[d.ts >= VAL_FROM].reset_index(drop=True)
seen = va.site_id.isin(tr.site_id.unique()).to_numpy()
k_tr = np.where(tr.hour.between(11, 15), 3.0, 1.0)
age_days = (VAL_FROM - tr.ts).dt.days.to_numpy()


def run(name, feats=BASE, weight=None, ratio=False, mask=None, **kw):
    p = dict(objective="regression", n_estimators=800, learning_rate=0.05, num_leaves=127,
             min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
             verbose=-1, n_jobs=8)
    p.update(kw)
    t = tr if mask is None else tr[mask]
    w = None if weight is None else weight[mask] if mask is not None else weight
    if ratio:
        den = np.maximum(t.csky.to_numpy(), 0.02)
        y = np.clip(t.cf.to_numpy() / den, 0, 2)
        w = (den if w is None else w * den) ** 2   # 원 스케일 제곱오차와 동일하게
        m = lgb.LGBMRegressor(**p).fit(t[feats], y, sample_weight=w)
        cf = m.predict(va[feats]) * np.maximum(va.csky.to_numpy(), 0.02)
    else:
        m = lgb.LGBMRegressor(**p).fit(t[feats], t.cf, sample_weight=w)
        cf = m.predict(va[feats])
    kwh = np.clip(cf, 0, None) * va.capacity_kw.to_numpy()
    kwh = np.where(va.elev.to_numpy() > -3, kwh, 0)
    r = score_groups(group_frame(va, kwh, grp), smp)
    r2 = score_groups(group_frame(va[seen], kwh[seen], grp), smp)
    log(f"{name:30s} all {r['pen']/1e6:8.3f}M | seen {r2['pen']/1e6:8.3f}M")
    np.save(f"../cache/val_{name}.npy", kwh)
    return m


which = sys.argv[1:] or ["wk", "ratio", "noid", "recent"]
if "wk" in which:
    run("weight_k", weight=k_tr)
if "ratio" in which:
    run("ratio_target", ratio=True)
if "noid" in which:
    run("no_site_id", feats=[c for c in BASE if c not in ("site_code", "lat", "lon")])
if "recent" in which:
    run("recency_w", weight=np.exp(-age_days / 730))
