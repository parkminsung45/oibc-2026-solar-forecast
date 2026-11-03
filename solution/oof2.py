"""구성 최적화용 OOF: 2023-06~2025-05를 주(week mod 4)로 4분할, 나머지 전 기간으로 학습.

사용: python oof2.py <config> [...]   → cache/oof2/{config}.npy (load_train 행 순서, 구간 밖 NaN)
"""
import os
import sys
import numpy as np
import pandas as pd
from common import TZ, log
from tune import CONFIGS, data, fit_predict

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "oof2")
os.makedirs(OUT, exist_ok=True)
LO = pd.Timestamp("2023-06-01 01:00", tz=TZ)
TRAIN_END = pd.Timestamp("2025-06-01 01:00", tz=TZ)


def span_mask(d):
    return (d.ts >= LO).to_numpy()


if __name__ == "__main__":
    D = data()
    d = D["d"]
    span = span_mask(d)
    fold = d.ts.dt.isocalendar().week.to_numpy() % 4
    day = (d.elev > -3).to_numpy()
    for name in sys.argv[1:]:
        path = f"{OUT}/{name}.npy"
        if os.path.exists(path):
            continue
        cfg = CONFIGS[name]
        feats = [c for c in D["feats"] if c not in cfg["drop"]]
        oof = np.full(len(d), np.nan)
        for f in range(4):
            te = span & (fold == f)
            oof[te] = fit_predict(cfg, d[day & ~te], d[te], feats, TRAIN_END)
            log(f"{name} fold {f}")
        np.save(path, oof)
