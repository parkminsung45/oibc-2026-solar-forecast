"""공간 평활 특징: 각 발전소의 근접 발전소 K개가 쓰는 관측지점 예보 평균.

좌표는 이웃 찾기에만 사용한다 (위치 특정 X). 결과는 feat_{split} 행 순서와 정렬된 별도 캐시.
"""
import os
import numpy as np
import pandas as pd
from features import CACHE, DATA, TZ, load_sites

VARS = ["wa_cloud", "wa_humidity", "wb_cloud", "wb_uv_idx", "wb_vis", "wb_rel_hum", "wb_precip_prob"]
K = 12


def neighbor_stations(sites: pd.DataFrame, col: str) -> dict:
    xy = np.c_[sites.lat.to_numpy(), sites.lon.to_numpy() * np.cos(np.radians(36))]
    dist = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(dist, axis=1)[:, :K]
    st = sites[col].to_numpy()
    return {sites.site_id.iat[i]: list(st[nn[i]]) for i in range(len(sites))}  # 중복 = 가중


def build(split: str) -> pd.DataFrame:
    path = f"{CACHE}/featnb_{split}.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    sites = load_sites()
    base = pd.read_parquet(f"{CACHE}/feat_{split}.parquet", columns=["site_id", "ts"])
    out = {}
    for prov, key in (("wa", "wa_station"), ("wb", "wb_station")):
        f = pd.read_parquet(f"{DATA}/{split}/forecast_{prov}.parquet")
        f["ts"] = pd.to_datetime(f.ts).dt.tz_convert(TZ)
        vs = [v for v in VARS if v.startswith(prov)]
        nb = neighbor_stations(sites, key)
        # site → station 가중 행렬
        stations = sorted(f[key].unique())
        sidx = {s: i for i, s in enumerate(stations)}
        W = np.zeros((len(sites), len(stations)))
        for i, s in enumerate(sites.site_id):
            for st in nb[s]:
                W[i, sidx[st]] += 1.0 / K
        for r in (10, 17):
            fr = f[f["round"] == r]
            for v in vs:
                M = fr.pivot_table(index="ts", columns=key, values=v).reindex(columns=stations)
                valid = M.notna().to_numpy().astype(float)
                num = np.nan_to_num(M.to_numpy()) @ W.T
                den = valid @ W.T
                S = pd.DataFrame(num / np.where(den > 0, den, np.nan), index=M.index,
                                 columns=sites.site_id).stack().rename(f"{v}_nb_r{r}")
                out[f"{v}_nb_r{r}"] = S
    nbdf = pd.concat(out.values(), axis=1).reset_index()
    nbdf.columns = ["ts", "site_id"] + list(out.keys())
    res = base.merge(nbdf, on=["site_id", "ts"], how="left").drop(columns=["site_id", "ts"])
    res = res.astype("float32")
    res.to_parquet(path, index=False)
    return res


if __name__ == "__main__":
    for s in ("train", "test"):
        r = build(s)
        print(s, r.shape, r.isna().mean().max())
