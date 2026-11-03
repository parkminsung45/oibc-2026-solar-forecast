"""검증 분할·채점 공통 함수."""
from __future__ import annotations
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from features import DATA, TZ, get, load_sites  # noqa: E402

VAL_FROM = pd.Timestamp("2024-06-01 01:00", tz=TZ)
VAL_TO = pd.Timestamp("2025-06-01 00:00", tz=TZ)

t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def load_train():
    d = get("train")
    d["cf"] = (d.energy / d.capacity_kw).clip(0, 1.1).astype("float32")
    return d


def train_smp() -> pd.DataFrame:
    s = pd.read_csv(f"{DATA}/train/smp.csv")
    s["ts"] = pd.to_datetime(s.ts).dt.tz_convert(TZ)
    s["k"] = np.where(s.ts.dt.hour.between(11, 15), 3, 1)
    return s


def score_groups(df: pd.DataFrame, smp: pd.DataFrame) -> dict:
    """df: group, ts, pred, actual (집합 단위). 반환: 대회 점수 S와 페널티 비율."""
    m = df.merge(smp, on="ts", how="inner")
    rev = (m.smp * m.actual).sum()
    pen = 0.10 * (m.smp * m.k * (m.pred - m.actual).abs()).sum()
    return {"S": rev - pen, "pen": pen, "pen_pct": 100 * pen / rev}


def group_frame(d: pd.DataFrame, pred_kwh: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    """발전소별 예측을 집합 합계로. groups: site_id → G1/G2."""
    t = pd.DataFrame({"group": d.site_id.map(groups).to_numpy(), "ts": d.ts.to_numpy(),
                      "pred": pred_kwh, "actual": d.energy.to_numpy()})
    return t.groupby(["group", "ts"], as_index=False)[["pred", "actual"]].sum()


def ref_groups():
    s = load_sites()
    return s.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
