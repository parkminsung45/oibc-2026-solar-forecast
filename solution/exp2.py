"""실험 2: 발전소 인코딩 특징 / 하이퍼파라미터 (참조 구성, 검증 2024-06~)."""
import sys
import numpy as np
import lightgbm as lgb
from common import VAL_FROM, load_train, train_smp, score_groups, group_frame, ref_groups, log
from siteenc import site_encodings, ENC_COLS

d = load_train()
smp = train_smp()
grp = ref_groups()
EXCL = {"site_id", "ts", "energy", "cf", "current_group", "wa_station", "wb_station"}
BASE = [c for c in d.columns if c not in EXCL]
tr = d[(d.ts < VAL_FROM) & (d.elev > -3)]
enc = site_encodings(tr)
d = d.merge(enc, on="site_id", how="left")
tr = d[(d.ts < VAL_FROM) & (d.elev > -3)]
va = d[d.ts >= VAL_FROM]
seen = va.site_id.isin(tr.site_id.unique())
log(f"인코딩 결측 발전소(검증 구간 신규): {va.groupby('site_id').site_eff.first().isna().sum()}")


def run(name, feats, **kw):
    p = dict(objective="regression", n_estimators=800, learning_rate=0.05, num_leaves=127,
             min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
             verbose=-1, n_jobs=8)
    p.update(kw)
    m = lgb.LGBMRegressor(**p).fit(tr[feats], tr.cf)
    kwh = np.clip(m.predict(va[feats]), 0, None) * va.capacity_kw.to_numpy()
    kwh = np.where(va.elev.to_numpy() > -3, kwh, 0)
    r = score_groups(group_frame(va, kwh, grp), smp)
    sm = seen.to_numpy()
    r2 = score_groups(group_frame(va[sm], kwh[sm], grp), smp)
    log(f"{name:34s} all {r['pen']/1e6:8.3f}M | seen {r2['pen']/1e6:8.3f}M")
    np.save(f"../cache/val_{name.replace(' ','_')}.npy", kwh)
    return m


which = sys.argv[1:] or ["base", "enc", "deep"]
if "base" in which:
    run("base", BASE)
if "enc" in which:
    run("base + site enc", BASE + ENC_COLS)
if "noid" in which:
    run("base + enc - site_code/latlon", [c for c in BASE + ENC_COLS if c not in ("site_code",)])
if "deep" in which:
    run("enc, lr0.03 n2000 leaves255", BASE + ENC_COLS, learning_rate=0.03, n_estimators=2000,
        num_leaves=255, min_child_samples=300)
