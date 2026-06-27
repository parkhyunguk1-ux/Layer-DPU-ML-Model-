"""
시계열 피처 엔지니어링
- 레이어별 DPU rolling 통계 (추세, 평균, 변화율)
- 다음 Glass 예측을 위한 lag 피처
"""

import numpy as np
import pandas as pd

LAYER_COLS = ["하판_RGB", "하판_PAC", "하판_CS", "하판_PI", "상판_BM", "상판_OC", "상판_PI"]
WINDOWS    = [10, 30, 100]   # rolling 윈도우 크기 (Glass 단위)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """레이어별 rolling mean, std, slope (추세 방향) 추가"""
    df = df.sort_values("timestamp").reset_index(drop=True)

    for layer in LAYER_COLS:
        for w in WINDOWS:
            roll = df[layer].rolling(w, min_periods=max(1, w // 2))
            df[f"{layer}_roll{w}_mean"] = roll.mean()
            df[f"{layer}_roll{w}_std"]  = roll.std().fillna(0)

            # 기울기 (slope): 양수 = 증가 추세, 음수 = 감소 추세
            df[f"{layer}_roll{w}_slope"] = (
                df[layer].rolling(w, min_periods=max(2, w // 2))
                .apply(_slope, raw=True)
            )

        # 단기(10) vs 장기(100) 평균 차이 → 최근 급등 여부
        df[f"{layer}_trend_signal"] = (
            df[f"{layer}_roll10_mean"] - df[f"{layer}_roll100_mean"]
        ).fillna(0)

    return df


def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """레이어 간 집계 파생 피처"""
    df["total_dpu"]       = df[LAYER_COLS].sum(axis=1)
    df["max_dpu"]         = df[LAYER_COLS].max(axis=1)
    df["mean_dpu"]        = df[LAYER_COLS].mean(axis=1)
    df["std_dpu"]         = df[LAYER_COLS].std(axis=1)
    df["nonzero_layers"]  = (df[LAYER_COLS] > 0).sum(axis=1)
    df["large_dpu_count"] = (df[LAYER_COLS] >= 50).sum(axis=1)

    lower = ["하판_RGB", "하판_PAC", "하판_CS", "하판_PI"]
    upper = ["상판_BM",  "상판_OC",  "상판_PI"]
    df["하판_total"] = df[lower].sum(axis=1)
    df["상판_total"] = df[upper].sum(axis=1)
    return df


def add_lag_features(df: pd.DataFrame, lags: list = [1, 3, 5]) -> pd.DataFrame:
    """직전 N개 Glass의 DPU 및 불량 여부 lag 피처"""
    for lag in lags:
        for layer in LAYER_COLS:
            df[f"{layer}_lag{lag}"] = df[layer].shift(lag)
        if "label" in df.columns:
            df[f"label_lag{lag}"] = df["label"].shift(lag)
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = add_aggregate_features(df)
    df = add_rolling_features(df)
    df = add_lag_features(df)

    # rolling/lag로 생긴 NaN 제거
    df = df.dropna().reset_index(drop=True)
    print(f"피처 엔지니어링 완료: {df.shape[0]:,}행 × {df.shape[1]}열")
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"timestamp", "glass_id", "label"}
    return [c for c in df.columns if c not in exclude]


def _slope(y: np.ndarray) -> float:
    """numpy 배열에서 선형 회귀 기울기 반환"""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    denom = (x * x).sum()
    return float((x * y).sum() / denom) if denom != 0 else 0.0
