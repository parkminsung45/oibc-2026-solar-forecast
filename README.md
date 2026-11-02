# OIBC 경진대회 2026 — 집합자원 구성 최적화 및 하루전 발전량 예측

- 대회 페이지: https://dataen.ai/challenge/2026
- 대회 기간: 2026-11-02 ~ 2026-11-13
- 과제: 태양광 발전소들을 두 개의 집합자원(VPP)으로 구성하고, 각 집합자원의 하루 전 발전량을 예측

## 구조
```
01_participant_dataset/
├── README.md      # 주최측 데이터 설명
├── baseline.py    # 주최측 베이스라인
├── train/         # (gitignore) 학습 데이터 parquet
└── test/          # (gitignore) 테스트 예보 데이터 parquet
```

원본 데이터(`*.parquet`, `sites.csv`, `smp.csv`)는 대회 규정 및 용량 문제로 저장소에 포함하지 않습니다.
대회 페이지에서 받은 데이터를 `01_participant_dataset/` 아래에 두고 사용하세요.
