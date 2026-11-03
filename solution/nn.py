"""MLP (앙상블 다양성용). GBDT와 같은 타깃(설비이용률/청천일사)·가중치를 쓴다.

수치 특징 → 표준화(결측 0) + 발전소 임베딩 → 3층 MLP. MPS(애플 GPU) 사용.
"""
import numpy as np
import torch
import torch.nn as nn

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CAT = "site_code"


class Net(nn.Module):
    def __init__(self, n_num, n_site, emb=16, h=512, p=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_site + 1, emb)
        self.f = nn.Sequential(
            nn.Linear(n_num + emb, h), nn.SiLU(), nn.Dropout(p),
            nn.Linear(h, h), nn.SiLU(), nn.Dropout(p),
            nn.Linear(h, h // 2), nn.SiLU(),
            nn.Linear(h // 2, 1))

    def forward(self, x, s):
        return nn.functional.softplus(self.f(torch.cat([x, self.emb(s)], 1))).squeeze(1)


def _prep(X, feats, stats=None):
    num = [c for c in feats if c != CAT]
    A = X[num].to_numpy(np.float32)
    if stats is None:
        mu, sd = np.nanmean(A, 0), np.nanstd(A, 0) + 1e-6
        stats = (mu, sd)
    A = np.nan_to_num((A - stats[0]) / stats[1], nan=0.0).clip(-8, 8).astype(np.float32)
    return A, X[CAT].to_numpy(np.int64), stats


def fit_predict(Xtr, y, w, Xva, feats, epochs=12, bs=4096, lr=2e-3, seed=0, **kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    A, s, stats = _prep(Xtr, feats)
    B, sv, _ = _prep(Xva, feats, stats)
    n_site = int(max(s.max(), sv.max()))
    net = Net(A.shape[1], n_site, **kw).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(A) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    w = (w / w.mean()).astype(np.float32)
    At, st = torch.from_numpy(A).to(DEV), torch.from_numpy(s).to(DEV)
    yt, wt = torch.from_numpy(y.astype(np.float32)).to(DEV), torch.from_numpy(w).to(DEV)
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(len(A), device=DEV)
        for i in range(len(A) // bs):
            idx = perm[i * bs:(i + 1) * bs]
            loss = (wt[idx] * (net(At[idx], st[idx]) - yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    out = []
    Bt, svt = torch.from_numpy(B).to(DEV), torch.from_numpy(sv).to(DEV)
    with torch.no_grad():
        for i in range(0, len(B), 65536):
            out.append(net(Bt[i:i + 65536], svt[i:i + 65536]).cpu().numpy())
    return np.concatenate(out)
