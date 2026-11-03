"""2-폴드 백테스트 실험 도구.

A: ~2023-05 학습 → 2023-06~2024-05 검증 / B: ~2024-05 학습 → 2024-06~2025-05 검증.
지표: 이력 있는 발전소(seen), 참조 구성의 집합 페널티. 검증 예측은 cache/cv/{name}_{fold}.npy.
사용: python tune.py <config> [<config> ...]   (CONFIGS 참고)
"""
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import TZ, load_train, train_smp, score_groups, group_frame, ref_groups, log

CV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "cv")
os.makedirs(CV, exist_ok=True)
FOLDS = {"A": (pd.Timestamp("2023-06-01 01:00", tz=TZ), pd.Timestamp("2024-06-01 01:00", tz=TZ)),
         "B": (pd.Timestamp("2024-06-01 01:00", tz=TZ), pd.Timestamp("2025-06-01 01:00", tz=TZ))}
EXCL = {"site_id", "ts", "energy", "cf", "current_group", "wa_station", "wb_station"}

LGB = dict(objective="regression", n_estimators=800, learning_rate=0.05, num_leaves=127,
           min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
           verbose=-1, n_jobs=8)
XGB = dict(n_estimators=1000, learning_rate=0.05, max_depth=9, min_child_weight=50,
           subsample=0.8, colsample_bytree=0.7, tree_method="hist", n_jobs=8)


def C(kind="lgb", decay=730, wk=1.0, drop=(), **params):
    return dict(kind=kind, decay=decay, wk=wk, drop=set(drop), params=params)


CONFIGS = {
    "lgb": C(),                                   # 현재 제출 설정
    "lgb_s1": C(random_state=1),
    "xgb": C("xgb"),
    "lgb_d365": C(decay=365),
    "lgb_d1460": C(decay=1460),
    "lgb_d0": C(decay=0),
    "lgb_wk2": C(wk=2.0),
    "lgb_wk3": C(wk=3.0),
    "lgb_nolatlon": C(drop=("lat", "lon", "wa_dist", "wb_dist")),
    "lgb_lr03": C(learning_rate=0.03, n_estimators=2000),
    "lgb_l255": C(num_leaves=255, min_child_samples=100),
    "lgb_l63": C(num_leaves=63, n_estimators=1600),
    "lgb_mcs1k": C(min_child_samples=1000),
    "lgb_ff4": C(colsample_bytree=0.4),
    "lgb_huber": C(objective="huber", alpha=0.3),
    "lgb_l2reg": C(reg_lambda=10.0, min_split_gain=0.0),
    "xgb_d11": C("xgb", max_depth=11, min_child_weight=100),
    "xgb_lr03": C("xgb", learning_rate=0.03, n_estimators=2000),
    "nn": C("nn"),
    "nn_e6": C("nn", epochs=6),
    "nn_s1": C("nn", seed=1),
    "nn_e24": C("nn", epochs=24),
    "nn_h1024": C("nn", h=1024, p=0.2),
    "nn_emb32": C("nn", emb=32, lr=1e-3, epochs=16),
    "nn_e3": C("nn", epochs=3),
    "nn_e4": C("nn", epochs=4),
    "nn_e6_s1": C("nn", epochs=6, seed=1),
    "nn_e6_s2": C("nn", epochs=6, seed=2),
    "nn_e6_p3": C("nn", epochs=6, p=0.3),
    "nn_e6_h1k": C("nn", epochs=6, h=1024, p=0.2),
    "nn_e4_bs16k": C("nn", epochs=4, bs=16384, lr=4e-3),
    "nn_e4_bs16k_s1": C("nn", epochs=4, bs=16384, lr=4e-3, seed=1),
    "nn_e4_bs16k_s2": C("nn", epochs=4, bs=16384, lr=4e-3, seed=2),
    "nn_e2_bs16k": C("nn", epochs=2, bs=16384, lr=4e-3),
    "nn_e3_bs16k": C("nn", epochs=3, bs=16384, lr=4e-3),
    "nn_e6_bs16k": C("nn", epochs=6, bs=16384, lr=4e-3),
    "nn_e4_bs32k": C("nn", epochs=4, bs=32768, lr=6e-3),
    "nn_e8_bs32k": C("nn", epochs=8, bs=32768, lr=6e-3),
    "nn_e4_bs16k_lr2": C("nn", epochs=4, bs=16384, lr=2e-3),
    "nn_e4_bs16k_p3": C("nn", epochs=4, bs=16384, lr=4e-3, p=0.3),
    "nn_e4_bs16k_p3_s1": C("nn", epochs=4, bs=16384, lr=4e-3, p=0.3, seed=1),
    "nn_e4_bs16k_p3_s2": C("nn", epochs=4, bs=16384, lr=4e-3, p=0.3, seed=2),
    "nn_e3_bs16k_s1": C("nn", epochs=3, bs=16384, lr=4e-3, seed=1),
}
# wa_ground_press는 평가 구간 2026-01 중순부터 일부 지점이 지면기압으로 바뀜(최저 924 hPa) → 제외 변형
PRESS = ("wa_ground_press_r10", "wa_ground_press_r17")
CONFIGS.update({f"{k}_np": {**v, "drop": v["drop"] | set(PRESS)} for k, v in list(CONFIGS.items())})


def make(cfg):
    if cfg["kind"] == "lgb":
        return lgb.LGBMRegressor(**{**LGB, **cfg["params"]})
    import xgboost as xgb
    return xgb.XGBRegressor(**{**XGB, **cfg["params"]})


def fit_predict(cfg, tr, va, feats, ref_time):
    den = np.maximum(tr.csky.to_numpy(), 0.02)
    y = np.clip(tr.cf.to_numpy() / den, 0, 2)
    w = den ** 2
    if cfg["decay"]:
        w = w * np.exp(-(ref_time - tr.ts).dt.days.to_numpy() / cfg["decay"])
    if cfg["wk"] != 1.0:
        w = w * np.where(tr.hour.between(11, 15), cfg["wk"], 1.0)
    if cfg["kind"] == "nn":
        import nn
        r = nn.fit_predict(tr, y, w, va, feats, **cfg["params"])
    else:
        r = make(cfg).fit(tr[feats], y, sample_weight=w).predict(va[feats])
    kwh = np.clip(r * np.maximum(va.csky.to_numpy(), 0.02), 0, None) * va.capacity_kw.to_numpy()
    return np.where(va.elev.to_numpy() > -3, kwh, 0.0)


_D = {}


def data():
    if not _D:
        d = load_train()
        _D.update(d=d, smp=train_smp(), grp=ref_groups(),
                  feats=[c for c in d.columns if c not in EXCL])
    return _D


def fold_frames(f):
    D = data()
    d = D["d"]
    lo, hi = FOLDS[f]
    tr = d[(d.ts < lo) & (d.elev > -3)]
    va = d[(d.ts >= lo) & (d.ts < hi)].reset_index(drop=True)
    seen = va.site_id.isin(tr.site_id.unique()).to_numpy()
    return tr, va, seen


_S = {}


def score(f, kwh):
    """seen 발전소·참조 구성 페널티(백만원). score_groups와 동일한 값을 빠르게."""
    if f not in _S:
        D = data()
        _, va, seen = fold_frames(f)
        key = pd.MultiIndex.from_arrays([va.site_id.map(D["grp"]).to_numpy(), va.ts])
        codes, uniq = pd.factorize(key)
        smp = D["smp"].set_index("ts")
        ts = uniq.get_level_values(1)
        wt = (smp.smp * smp.k).reindex(ts).to_numpy()
        ok = ~np.isnan(wt)
        act = np.bincount(codes[seen], va.energy.to_numpy()[seen], len(uniq))
        _S[f] = (codes, seen, act, np.nan_to_num(wt), ok)
    codes, seen, act, wt, ok = _S[f]
    pred = np.bincount(codes[seen], kwh[seen], len(act))
    return 0.10 * float((wt * np.abs(pred - act))[ok].sum()) / 1e6


def run(name):
    cfg = CONFIGS[name]
    D = data()
    feats = [c for c in D["feats"] if c not in cfg["drop"]]
    out = {}
    for f in FOLDS:
        path = f"{CV}/{name}_{f}.npy"
        if os.path.exists(path):
            kwh = np.load(path)
        else:
            tr, va, _ = fold_frames(f)
            kwh = fit_predict(cfg, tr, va, feats, FOLDS[f][0])
            np.save(path, kwh)
        out[f] = score(f, kwh)
    log(f"{name:14s} A {out['A']:8.3f}M  B {out['B']:8.3f}M  합 {out['A']+out['B']:8.3f}M")
    return out


if __name__ == "__main__":
    for n in sys.argv[1:] or list(CONFIGS):
        run(n)
