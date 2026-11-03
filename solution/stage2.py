"""2단계: 집합 단위 보정 + 구성 최적화 검증.

- OOF 발전소별 예측을 집합 합계로 묶어, 집합 단위 L1(가중치 SMP·k) 잔차 모델로 보정
- 구성: 참조 / 무작위 / 잔차 기반 국소탐색 최적화 비교
학습 B0~B2, 평가 B3(2024-06~2025-05).
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import load_sites, train_smp, ref_groups, log
from features import CACHE, get

OBJ = sys.argv[1] if len(sys.argv) > 1 else "regression"
oof = pd.read_parquet(f"{CACHE}/oof_{OBJ}.parquet")
sites = load_sites().set_index("site_id")
cap = sites.capacity_kw
smp = train_smp()
oof = oof.merge(smp[["ts", "smp", "k"]], on="ts", how="inner")
oof["cap"] = oof.site_id.map(cap)
oof["w"] = oof.smp * oof.k

# 집합 공통 기상 특징 (시각별 지역평균) — 특징 테이블에서 시각당 하나
feat = get("train")
REG = [c for c in feat.columns if c.endswith("_reg_r17") or c.endswith("_reg_r10")]
reg = feat.groupby("ts")[REG + ["csky", "elev"]].mean().reset_index()
del feat


def group_table(groups: pd.Series) -> pd.DataFrame:
    t = oof.assign(group=oof.site_id.map(groups))
    a = t.groupby(["group", "ts"], as_index=False).agg(
        pred=("pred_kwh", "sum"), actual=("energy", "sum"), capr=("cap", "sum"),
        block=("block", "first"), smp=("smp", "first"), k=("k", "first"))
    a = a.merge(reg, on="ts", how="left")
    a["hour"] = a.ts.dt.hour
    doy = a.ts.dt.dayofyear
    a["doy_sin"], a["doy_cos"] = np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)
    a["pcf"] = a.pred / a.capr
    a["gid"] = (a.group == "G2").astype(int)
    return a


S2_FEATS = ["pcf", "hour", "doy_sin", "doy_cos", "csky", "elev", "gid"] + REG


def penalty(a, col):
    return 0.10 * (a.smp * a.k * (a[col] - a.actual).abs()).sum()


def calibrate(a: pd.DataFrame, train_mask, n=300):
    """잔차(actual-pred)/capr 를 L1·가중치(SMP·k·capr)로 학습."""
    tr = a[train_mask & (a.elev > -3)]
    y = (tr.actual - tr.pred) / tr.capr
    m = lgb.LGBMRegressor(objective="l1", n_estimators=n, learning_rate=0.03, num_leaves=31,
                          min_child_samples=100, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, verbose=-1, n_jobs=8)
    m.fit(tr[S2_FEATS], y, sample_weight=tr.smp * tr.k * tr.capr + 1e-3)
    adj = m.predict(a[S2_FEATS]) * a.capr
    return np.where(a.elev > -3, np.clip(a.pred + adj, 0, None), 0), m


def evaluate(name, groups):
    a = group_table(groups)
    te = (a.block == 3).to_numpy()
    raw = penalty(a[te], "pred")
    a["cal"], _ = calibrate(a, ~te)
    cal = penalty(a[te], "cal")
    tot = a[te].groupby("ts").agg(p=("pred", "sum"), y=("actual", "sum"), smp=("smp", "first"), k=("k", "first"))
    one = 0.10 * (tot.smp * tot.k * (tot.p - tot.y).abs()).sum()
    log(f"{name:22s} raw {raw/1e6:8.3f}M  calibrated {cal/1e6:8.3f}M  (단일집합 하한 {one/1e6:.3f}M)")
    return cal


if __name__ == "__main__":
    ref = ref_groups()
    evaluate("reference A/B", ref)
    rng = np.random.default_rng(0)
    for i in range(2):
        ids = cap.index.to_numpy()
        while True:
            g = pd.Series(np.where(rng.random(len(ids)) < 0.5, "G1", "G2"), index=ids)
            c1 = cap[g == "G1"].sum()
            if 20000 <= c1 <= cap.sum() - 20000:
                break
        evaluate(f"random #{i}", g)
