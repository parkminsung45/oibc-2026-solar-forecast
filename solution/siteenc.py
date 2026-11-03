"""발전소 고유 특성(타깃 인코딩). 반드시 학습 구간 데이터로만 계산한다."""
import numpy as np
import pandas as pd


def site_encodings(tr: pd.DataFrame) -> pd.DataFrame:
    """tr: 학습 행 (site_id, hour, cf, csky, wa_cloud_r17, wb_cloud_r17 필요)."""
    clear = tr[(tr.wa_cloud_r17 < 20) & (tr.wb_cloud_r17 < 20) & (tr.csky > 0.2)]
    eff = (clear.cf / clear.csky).groupby(clear.site_id).median().rename("site_eff")
    # 시간대 형태: 전체 발전소 평균 대비 발전소 시간대별 비율
    t = tr[tr.csky > 0.05]
    hm = t.groupby(["site_id", "hour"]).cf.mean().unstack()
    allm = t.groupby("hour").cf.mean()
    rel = hm / allm
    mid = rel[[11, 12, 13]].mean(axis=1)
    out = pd.DataFrame({
        "site_eff": eff,
        "site_am": rel[[7, 8, 9]].mean(axis=1) / mid,
        "site_pm": rel[[15, 16, 17]].mean(axis=1) / mid,
        "site_mid": mid,
        "site_p95": t.groupby("site_id").cf.quantile(0.95),
        "site_n": t.groupby("site_id").size().astype(float),
    })
    return out.astype("float32").reset_index()


ENC_COLS = ["site_eff", "site_am", "site_pm", "site_mid", "site_p95", "site_n"]
