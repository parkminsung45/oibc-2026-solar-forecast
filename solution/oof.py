"""연도 블록 K-fold로 발전소별 OOF 예측을 만든다 (2단계 보정·구성 최적화용).

블록: B0 ~2022-05, B1 2022-06~2023-05, B2 2023-06~2024-05, B3 2024-06~2025-05
B3 예측은 B0~B2로만 학습 → 정직한 검증. 다른 블록은 미래 블록도 학습에 쓴다.
출력: cache/oof_{obj}.parquet (site_id, ts, block, pred_kwh)
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import TZ, load_train, log

OBJ = sys.argv[1] if len(sys.argv) > 1 else "regression"
EDGES = [pd.Timestamp(f"{y}-06-01 01:00", tz=TZ) for y in (2022, 2023, 2024)]
PARAMS = dict(n_estimators=800, learning_rate=0.05, num_leaves=127, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=8)

d = load_train()
d["block"] = np.searchsorted(np.array(EDGES, dtype="datetime64[ns]"),
                             d.ts.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(), side="right")
EXCL = {"site_id", "ts", "energy", "cf", "current_group", "wa_station", "wb_station", "block"}
FEATS = [c for c in d.columns if c not in EXCL]
day = d.elev.to_numpy() > -3
out = np.zeros(len(d), dtype="float32")
for b in range(4):
    trm = (d.block != b).to_numpy() & day
    tem = (d.block == b).to_numpy()
    m = lgb.LGBMRegressor(objective=OBJ, **PARAMS).fit(d.loc[trm, FEATS], d.cf[trm])
    p = np.clip(m.predict(d.loc[tem, FEATS]), 0, None) * d.capacity_kw[tem].to_numpy()
    out[tem] = np.where(day[tem], p, 0)
    log(f"block {b}: train {trm.sum():,} → pred {tem.sum():,}")
res = d[["site_id", "ts", "block", "energy"]].copy()
res["pred_kwh"] = out
res.to_parquet(f"../cache/oof_{OBJ}.parquet", index=False)
log("saved")
