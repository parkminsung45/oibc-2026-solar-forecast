"""공통 특징 생성. (site, ts) 격자에 하루전 예보(라운드 10·17 모두)·태양기하·지역평균을 붙인다.

캐시: cache/feat_train.parquet, cache/feat_test.parquet
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{ROOT}/01_participant_dataset"
CACHE = f"{ROOT}/cache"
TZ = "Asia/Seoul"

WA_COLS = ["wa_cloud", "wa_temp", "wa_humidity", "wa_ground_press",
           "wa_wind_speed", "wa_rain", "wa_snow"]
WB_COLS = ["wb_temp", "wb_dew_point", "wb_wind_speed", "wb_wind_gust_spd", "wb_rel_hum",
           "wb_vis", "wb_ceiling", "wb_uv_idx", "wb_precip_prob", "wb_snow_prob",
           "wb_total_liq", "wb_snow", "wb_cloud"]
# 시간 맥락(t±1, 일평균)을 붙일 핵심 변수
CTX_COLS = ["wa_cloud", "wb_cloud", "wb_uv_idx", "wb_vis", "wb_ceiling",
            "wb_precip_prob", "wa_humidity", "wb_rel_hum"]


def solar(ts: pd.DatetimeIndex, lat, lon):
    """태양 고도·방위 (NOAA 근사). ts 우측닫힘 → 구간 중앙(-30분)."""
    doy = ts.dayofyear.to_numpy()
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868*np.cos(g) - 0.032077*np.sin(g)
                       - 0.014615*np.cos(2*g) - 0.040849*np.sin(2*g))
    decl = (0.006918 - 0.399912*np.cos(g) + 0.070257*np.sin(g)
            - 0.006758*np.cos(2*g) + 0.000907*np.sin(2*g)
            - 0.002697*np.cos(3*g) + 0.00148*np.sin(3*g))
    tst = (hour - 0.5) * 60 + eqtime + 4 * (lon - 135.0)
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(lat)
    cosz = np.clip(np.sin(la)*np.sin(decl) + np.cos(la)*np.cos(decl)*np.cos(ha), -1, 1)
    elev = 90.0 - np.degrees(np.arccos(cosz))
    az = np.degrees(np.arctan2(-np.sin(ha),
                               np.tan(decl)*np.cos(la) - np.sin(la)*np.cos(ha))) % 360
    e0 = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    return elev, az, cosz, e0


def _hourly_solar_mean(ts, lat, lon, n=6):
    """한 시간 구간 내 n점 평균 cos(천정각)+ — 구간 적분 청천일사 근사."""
    acc = np.zeros(len(ts))
    for i in range(n):
        off = pd.Timedelta(minutes=60 * (i + 0.5) / n - 30)
        _, _, c, _ = solar(ts + off, lat, lon)
        acc += np.clip(c, 0, None) * np.exp(-0.057 / np.clip(c, 0.05, None)) * (c > 0)
    return acc / n


def prep_forecast(df: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    """라운드 10·17을 옆으로 펼치고 시간 맥락·일평균·지역평균 특징을 만든다."""
    df = df[[key, "round", "ts"] + cols].copy()
    df["ts"] = pd.to_datetime(df.ts).dt.tz_convert(TZ)
    df = df.sort_values([key, "round", "ts"])
    ctx = [c for c in CTX_COLS if c in cols]
    g = df.groupby([key, "round"], sort=False)
    for c in ctx:
        df[f"{c}_m1"] = g[c].shift(1)
        df[f"{c}_p1"] = g[c].shift(-1)
    day = (df.ts - pd.Timedelta(hours=1)).dt.date
    hr = df.ts.dt.hour
    daymask = (hr >= 8) & (hr <= 17)
    for c in ctx:
        tmp = df[c].where(daymask)
        df[f"{c}_dmean"] = tmp.groupby([df[key], df["round"], day]).transform("mean")
    # 지역평균 (같은 시각 전 관측지점)
    for c in ctx:
        df[f"{c}_reg"] = df.groupby(["round", "ts"])[c].transform("mean")
    feat = [c for c in df.columns if c not in (key, "round", "ts")]
    wide = df.pivot_table(index=[key, "ts"], columns="round", values=feat, aggfunc="first")
    wide.columns = [f"{c}_r{r}" for c, r in wide.columns]
    wide = wide.reset_index()
    # 라운드 간 차이 (예보 갱신량)
    for c in ctx:
        wide[f"{c}_d1710"] = wide[f"{c}_r17"] - wide[f"{c}_r10"]
    return wide.astype({c: "float32" for c in wide.columns if c not in (key, "ts")})


def build(split: str, sites: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    fa = prep_forecast(pd.read_parquet(f"{DATA}/{split}/forecast_wa.parquet"), "wa_station", WA_COLS)
    fb = prep_forecast(pd.read_parquet(f"{DATA}/{split}/forecast_wb.parquet"), "wb_station", WB_COLS)
    d = base.merge(sites, on="site_id", how="left")
    d = d.merge(fa, on=["wa_station", "ts"], how="left")
    d = d.merge(fb, on=["wb_station", "ts"], how="left")
    ts = pd.DatetimeIndex(d.ts)
    lat, lon = d.lat.to_numpy(), d.lon.to_numpy()
    elev, az, cosz, e0 = solar(ts, lat, lon)
    d["elev"], d["azim"], d["cosz"] = elev.astype("float32"), az.astype("float32"), cosz.astype("float32")
    d["csky"] = (_hourly_solar_mean(ts, lat, lon) * e0).astype("float32")
    doy = ts.dayofyear.to_numpy()
    d["doy_sin"] = np.sin(2*np.pi*doy/365.25).astype("float32")
    d["doy_cos"] = np.cos(2*np.pi*doy/365.25).astype("float32")
    d["hour"] = ts.hour.to_numpy().astype("int8")
    d["site_code"] = d.site_id.str[2:].astype("int16")
    # 청천일사 × (1-운량) 상호작용
    for r in (10, 17):
        d[f"csky_wa_r{r}"] = (d.csky * (1 - d[f"wa_cloud_r{r}"] / 100)).astype("float32")
        d[f"csky_wb_r{r}"] = (d.csky * (1 - d[f"wb_cloud_r{r}"] / 100)).astype("float32")
    return d


def load_sites():
    return pd.read_csv(f"{DATA}/sites.csv")


def get(split: str) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/feat_{split}.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    sites = load_sites()
    if split == "train":
        base = pd.read_parquet(f"{DATA}/train/generation.parquet")
        base["ts"] = pd.to_datetime(base.ts).dt.tz_convert(TZ)
    else:
        index = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(f"{DATA}/smp.csv").ts)).tz_convert(TZ)
        base = pd.MultiIndex.from_product([sites.site_id, index],
                                          names=["site_id", "ts"]).to_frame(index=False)
    base = base[base.ts.dt.hour.between(5, 20)]   # 야간(21~4시)은 0으로 고정
    d = build(split, sites, base)
    d.to_parquet(path, index=False)
    return d


if __name__ == "__main__":
    import time
    for s in ("train", "test"):
        t = time.time(); d = get(s)
        print(s, d.shape, f"{time.time()-t:.0f}s", d.isna().mean().sort_values().tail(3).to_dict())
