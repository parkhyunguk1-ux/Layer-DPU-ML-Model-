"""
MLOps 모니터링
- 레이어별 DPU 증가 추세 감지 (드리프트)
- 시간 구간별 레이어 영향도 변화 추적
- 모델 성능 열화 감지 → 재학습 트리거
"""

import json
import os
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.feature_engineering import LAYER_COLS, WINDOWS

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRIFT_SLOPE_THRESHOLD  = 0.05   # 레이어 DPU 기울기 이 값 이상이면 증가 추세 경고
RETRAIN_AUC_DROP       = 0.03   # 베이스라인 대비 AUC 하락 이 값 이상이면 재학습 권고


def analyze_dpu_trend(df: pd.DataFrame, freq: str = "1D") -> pd.DataFrame:
    """
    시간 구간별 레이어 평균 DPU 집계 + 선형 추세(slope) 계산
    freq: pandas resample 주기 ('1D'=일별, '1H'=시간별, '1W'=주별)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    # 구간별 평균 DPU
    resampled = df[LAYER_COLS].resample(freq).mean().dropna()

    # 각 레이어의 선형 추세 (전체 기간)
    trend_report = []
    for layer in LAYER_COLS:
        y = resampled[layer].values
        x = np.arange(len(y))
        slope, intercept, r, p, se = stats.linregress(x, y)
        status = "🔴 증가" if slope > DRIFT_SLOPE_THRESHOLD else (
                 "🟡 안정" if abs(slope) <= DRIFT_SLOPE_THRESHOLD else "🟢 감소")
        trend_report.append({
            "layer":     layer,
            "slope":     round(slope, 4),
            "r_squared": round(r**2, 4),
            "p_value":   round(p, 4),
            "status":    status,
        })

    trend_df = pd.DataFrame(trend_report).sort_values("slope", ascending=False)
    return resampled, trend_df


def plot_dpu_trend(resampled: pd.DataFrame, trend_df: pd.DataFrame):
    """레이어별 DPU 시계열 + 추세선 그래프"""
    n = len(LAYER_COLS)
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    axes = axes.flatten()

    for i, layer in enumerate(LAYER_COLS):
        ax = axes[i]
        y = resampled[layer].values
        x_num = np.arange(len(y))

        ax.plot(resampled.index, y, alpha=0.7, label="실제 DPU 평균")

        # 추세선
        slope = trend_df[trend_df["layer"] == layer]["slope"].values[0]
        trend_line = slope * x_num + y[0]
        color = "red" if slope > DRIFT_SLOPE_THRESHOLD else "gray"
        ax.plot(resampled.index, trend_line, "--", color=color, linewidth=1.5, label=f"추세 (slope={slope:.3f})")

        status = trend_df[trend_df["layer"] == layer]["status"].values[0]
        ax.set_title(f"{layer}  {status}", fontsize=10)
        ax.set_ylabel("평균 DPU")
        ax.legend(fontsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # 마지막 셀에 요약 테이블
    axes[-1].axis("off")
    table_data = trend_df[["layer", "slope", "status"]].values.tolist()
    table = axes[-1].table(
        cellText=table_data,
        colLabels=["레이어", "기울기", "상태"],
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    axes[-1].set_title("레이어별 DPU 추세 요약", fontsize=10)

    plt.suptitle("레이어별 DPU 시계열 추세 분석", fontsize=14, y=1.01)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/dpu_trend.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"추세 그래프 저장: {path}")


def detect_performance_drift(current_auc: float, baseline_auc: float) -> dict:
    """모델 AUC 열화 감지"""
    drop = baseline_auc - current_auc
    retrain_needed = drop >= RETRAIN_AUC_DROP
    result = {
        "baseline_auc":    round(baseline_auc, 4),
        "current_auc":     round(current_auc, 4),
        "auc_drop":        round(drop, 4),
        "retrain_needed":  retrain_needed,
        "status":          "🔴 재학습 권고" if retrain_needed else "🟢 정상",
    }
    print("\n[모델 성능 모니터링]")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


def run_monitoring(csv_path: str = "data.csv", freq: str = "1D"):
    print("=" * 55)
    print("  DPU 드리프트 + 레이어 영향도 모니터링")
    print("=" * 55)

    df = pd.read_csv(csv_path)
    resampled, trend_df = analyze_dpu_trend(df, freq=freq)

    print("\n[레이어별 DPU 증가 추세]")
    print(trend_df.to_string(index=False))

    plot_dpu_trend(resampled, trend_df)

    # 경고 레이어 출력
    warning_layers = trend_df[trend_df["slope"] > DRIFT_SLOPE_THRESHOLD]["layer"].tolist()
    if warning_layers:
        print(f"\n⚠️  DPU 증가 추세 감지: {warning_layers}")
        print("   → 해당 레이어 공정 점검 권고")
    else:
        print("\n✅ 모든 레이어 DPU 안정 상태")

    trend_df.to_csv(f"{OUTPUT_DIR}/trend_report.csv", index=False)
    return trend_df


if __name__ == "__main__":
    run_monitoring()
