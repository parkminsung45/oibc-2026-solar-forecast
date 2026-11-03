"""실험 4: ratio 타깃 + 최근성 가중 (+공간평활 특징), 모델 계열 다양화."""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import VAL_FROM, load_train, train_smp, score_groups, group_frame, ref_groups, log
from nbfeat import build as nb_build

d = load_train()
nb = nb_build("train")
d = pd.concat([d, nb], axis=1)
smp = train_smp()
grp = ref_groups()
EXCL = {"site_id", "ts", "energy", "cf", "current_group", "wa_station", "wb_station"}
NB = list(nb.columns)
BASE = [c for c in d.columns if c not in EXCL and c not in NB]
tr = d[(d.ts < VAL_FROM) & (d.elev > -3)]
va = d[d.ts >= VAL_FROM].reset_index(drop=True)
seen = va.site_id.isin(tr.site_id.unique()).to_numpy()
age = (VAL_FROM - tr.ts).dt.days.to_numpy()
den_tr = np.maximum(tr.csky.to_numpy(), 0.02)
den_va = np.maximum(va.csky.to_numpy(), 0.02)
y = np.clip(tr.cf.to_numpy() / den_tr, 0, 2)
w = np.exp(-age / 730) * den_tr ** 2


def finish(name, ratio_pred):
    kwh = np.clip(ratio_pred * den_va, 0, None) * va.capacity_kw.to_numpy()
    kwh = np.where(va.elev.to_numpy() > -3, kwh, 0)
    r = score_groups(group_frame(va, kwh, grp), smp)
    r2 = score_groups(group_frame(va[seen], kwh[seen], grp), smp)
    log(f"{name:30s} all {r['pen']/1e6:8.3f}M | seen {r2['pen']/1e6:8.3f}M")
    np.save(f"../cache/val_{name}.npy", kwh)


def lgbm(feats, seed=0, **kw):
    p = dict(objective="regression", n_estimators=800, learning_rate=0.05, num_leaves=127,
             min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
             verbose=-1, n_jobs=8, random_state=seed)
    p.update(kw)
    return lgb.LGBMRegressor(**p).fit(tr[feats], y, sample_weight=w).predict(va[feats])


which = sys.argv[1:] or ["lgb", "lgbnb", "xgb", "cat"]
F = BASE
if "lgb" in which:
    finish("lgb_ratio_rec", lgbm(BASE))
if "lgbnb" in which:
    finish("lgb_ratio_rec_nb", lgbm(F))
if "xgb" in which:
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=9, min_child_weight=50,
                         subsample=0.8, colsample_bytree=0.7, tree_method="hist", n_jobs=8)
    m.fit(tr[F], y, sample_weight=w)
    finish("xgb_ratio_rec", m.predict(va[F]))
if "cat" in which:
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=2000, learning_rate=0.08, depth=8, verbose=0, thread_count=8)
    m.fit(tr[F], y, sample_weight=w)
    finish("cat_ratio_rec", m.predict(va[F]))
