"""전 기간(~2025-05) 학습 → 평가 구간 발전소별 예측. cache/test2/{name}.npy (있으면 재사용).

사용: python testpred.py <config> [...]
"""
import os
import sys
import numpy as np
from common import log
from features import get
from tune import CONFIGS, data, fit_predict
from oof2 import TRAIN_END

TEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "test2")
os.makedirs(TEST, exist_ok=True)
_TE = {}


def test_frame():
    if not _TE:
        _TE["te"] = get("test")
    return _TE["te"]


def predict(name):
    path = f"{TEST}/{name}.npy"
    if not os.path.exists(path):
        D = data()
        d = D["d"]
        cfg = CONFIGS[name]
        feats = [c for c in D["feats"] if c not in cfg["drop"]]
        np.save(path, fit_predict(cfg, d[(d.elev > -3).to_numpy()], test_frame(), feats, TRAIN_END))
    p = np.load(path)
    log(f"{name}: test 합계 {p.sum()/1e6:.2f} GWh")
    return p


if __name__ == "__main__":
    for n in sys.argv[1:]:
        predict(n)
