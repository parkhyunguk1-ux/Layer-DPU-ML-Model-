"""
시계열 샘플 데이터 생성기
timestamp + glass_id + 레이어별 DPU + label
※ 시간이 지남에 따라 특정 레이어(하판_PI, 상판_BM)의 DPU가 점진적으로 증가하는 패턴 포함
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 100_000
DPU_SIZES = [0, 15, 30, 50, 80, 100]

start = pd.Timestamp("2024-01-01")
timestamps = pd.date_range(start, periods=N, freq="1min")
glass_ids  = [f"GL{str(i).zfill(6)}" for i in range(N)]

# 시간 인덱스 (0~1로 정규화) — 후반부일수록 특정 레이어 DPU 증가
t = np.linspace(0, 1, N)

def sample_dpu_with_trend(n, trend_strength=0.0):
    """trend_strength > 0 이면 시간이 지날수록 큰 DPU 값 확률 증가"""
    results = []
    for i in range(n):
        bias = trend_strength * t[i]
        weights = np.array([0.65 - bias*0.3, 0.15, 0.10, 0.06 + bias*0.1, 0.03 + bias*0.1, 0.01 + bias*0.1])
        weights = np.clip(weights, 0.01, 1.0)
        weights /= weights.sum()
        results.append(np.random.choice(DPU_SIZES, p=weights))
    return np.array(results)

def sample_dpu_stable(n):
    weights = np.array([0.65, 0.15, 0.10, 0.06, 0.03, 0.01])
    return np.random.choice(DPU_SIZES, size=n, p=weights)

# 레이어별 DPU (하판_PI, 상판_BM은 시간에 따라 증가 추세)
rgb      = sample_dpu_stable(N)
pac      = sample_dpu_stable(N)
cs       = sample_dpu_stable(N)
pi_lower = sample_dpu_with_trend(N, trend_strength=0.8)   # ← 증가 레이어
bm       = sample_dpu_with_trend(N, trend_strength=0.6)   # ← 증가 레이어
oc       = sample_dpu_stable(N)
pi_upper = sample_dpu_stable(N)

# 불량 확률 계산
total    = rgb + pac + cs + pi_lower + bm + oc + pi_upper
max_dpu  = np.stack([rgb, pac, cs, pi_lower, bm, oc, pi_upper], axis=1).max(axis=1)
nonzero  = np.stack([rgb, pac, cs, pi_lower, bm, oc, pi_upper], axis=1).astype(bool).sum(axis=1)

logit = -4.5 + 0.02 * total + 0.01 * max_dpu + 0.3 * nonzero
prob  = 1 / (1 + np.exp(-logit))
label = (np.random.rand(N) < prob).astype(int)

df = pd.DataFrame({
    "timestamp": timestamps,
    "glass_id":  glass_ids,
    "하판_RGB":   rgb,
    "하판_PAC":   pac,
    "하판_CS":    cs,
    "하판_PI":    pi_lower,
    "상판_BM":    bm,
    "상판_OC":    oc,
    "상판_PI":    pi_upper,
    "label":      label,
})

df.to_csv("data.csv", index=False)
print(f"샘플 데이터 생성 완료: {len(df):,}행")
print(f"불량률: {label.mean():.2%}  (불량={label.sum():,} / 정상={(label==0).sum():,})")
print(df.head())
