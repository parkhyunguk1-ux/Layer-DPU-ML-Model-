"""
샘플 데이터 생성기
실제 CSV가 없을 때 테스트용으로 사용
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 100_000

DPU_SIZES = [0, 15, 30, 50, 80, 100]  # 0 = 결함 없음

# 레이어별 DPU 샘플링 (불량일수록 큰 DPU 값 경향)
def sample_dpu(n, defect_bias=1.0):
    weights = np.array([0.6, 0.15, 0.1, 0.08, 0.05, 0.02])
    weights = weights ** (1 / defect_bias)
    weights /= weights.sum()
    return np.random.choice(DPU_SIZES, size=n, p=weights)

# 하판 레이어
rgb = sample_dpu(N)
pac = sample_dpu(N)
cs  = sample_dpu(N)
pi_lower = sample_dpu(N)

# 상판 레이어
bm = sample_dpu(N)
oc = sample_dpu(N)
pi_upper = sample_dpu(N)

# 불량 확률: 전체 DPU 합이 클수록, 큰 DPU 레이어가 많을수록 높아짐
total = rgb + pac + cs + pi_lower + bm + oc + pi_upper
max_dpu = np.stack([rgb, pac, cs, pi_lower, bm, oc, pi_upper], axis=1).max(axis=1)
nonzero = np.stack([rgb, pac, cs, pi_lower, bm, oc, pi_upper], axis=1).astype(bool).sum(axis=1)

logit = -4.5 + 0.02 * total + 0.01 * max_dpu + 0.3 * nonzero
prob = 1 / (1 + np.exp(-logit))
label = (np.random.rand(N) < prob).astype(int)

df = pd.DataFrame({
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
