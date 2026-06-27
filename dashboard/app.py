"""
Streamlit 대시보드
- 레이어별 DPU 시계열 추세 (증가 레이어 강조)
- 레이어 영향도 변화 (SHAP)
- 다음 Glass 불량 예측
- 모델 성능 모니터링
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.feature_engineering import LAYER_COLS
from src.monitor import analyze_dpu_trend, DRIFT_SLOPE_THRESHOLD

st.set_page_config(
    page_title="LCD DPU 불량 예측 MLOps",
    page_icon="🔬",
    layout="wide",
)

# ── 사이드바 ──────────────────────────────────────────────
st.sidebar.title("🔬 LCD DPU MLOps")
csv_path = st.sidebar.text_input("CSV 파일 경로", value="data.csv")
freq     = st.sidebar.selectbox("시간 집계 단위", ["1h", "6h", "1D", "1W"], index=2)
n_future = st.sidebar.slider("예측 구간 (Glass 수)", 10, 500, 100)

@st.cache_data
def load_data(path):
    return pd.read_csv(path, parse_dates=["timestamp"])

try:
    df = load_data(csv_path)
except FileNotFoundError:
    st.error(f"파일을 찾을 수 없습니다: {csv_path}")
    st.stop()

# ── 타이틀 ───────────────────────────────────────────────
st.title("🔬 LCD 레이어별 DPU 추세 & 불량 예측 MLOps")
st.caption(f"데이터: {len(df):,}건 | 기간: {df['timestamp'].min().date()} ~ {df['timestamp'].max().date()}")

# ── KPI 카드 ─────────────────────────────────────────────
resampled, trend_df = analyze_dpu_trend(df, freq=freq)
warning_layers = trend_df[trend_df["slope"] > DRIFT_SLOPE_THRESHOLD]["layer"].tolist()

col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 Glass", f"{len(df):,}개")
col2.metric("불량률", f"{df['label'].mean():.2%}")
col3.metric("증가 추세 레이어", f"{len(warning_layers)}개",
            delta=f"{warning_layers[0] if warning_layers else '없음'}",
            delta_color="inverse")
col4.metric("최근 불량률 (최근 1000건)",
            f"{df['label'].tail(1000).mean():.2%}",
            delta=f"{df['label'].tail(1000).mean() - df['label'].mean():.2%}",
            delta_color="inverse")

st.divider()

# ── 탭 ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 DPU 추세 분석",
    "🎯 레이어 영향도 (SHAP)",
    "🔮 다음 Glass 예측",
    "🚨 드리프트 모니터링",
])

# ── Tab 1: DPU 추세 ──────────────────────────────────────
with tab1:
    st.subheader("레이어별 DPU 시계열 추세")

    selected_layers = st.multiselect(
        "레이어 선택", LAYER_COLS, default=LAYER_COLS
    )

    fig = go.Figure()
    for layer in selected_layers:
        slope = trend_df[trend_df["layer"] == layer]["slope"].values[0]
        is_increasing = slope > DRIFT_SLOPE_THRESHOLD
        color = "red" if is_increasing else "steelblue"
        dash  = "solid"

        fig.add_trace(go.Scatter(
            x=resampled.index,
            y=resampled[layer],
            name=f"{layer} (slope={slope:.3f})",
            line=dict(color=color, dash=dash),
            mode="lines",
        ))

        # 추세선
        y = resampled[layer].values
        x_num = np.arange(len(y))
        trend_line = slope * x_num + y[0]
        fig.add_trace(go.Scatter(
            x=resampled.index,
            y=trend_line,
            name=f"{layer} 추세",
            line=dict(color=color, dash="dash", width=1),
            showlegend=False,
        ))

    fig.update_layout(
        title="레이어별 DPU 추세 (빨간색 = 증가 경고)",
        xaxis_title="시간",
        yaxis_title="평균 DPU",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("레이어별 추세 요약")
    styled = trend_df.style.apply(
        lambda row: ["background-color: #ffcccc" if row["slope"] > DRIFT_SLOPE_THRESHOLD
                     else "" for _ in row],
        axis=1
    )
    st.dataframe(styled, use_container_width=True)

# ── Tab 2: SHAP 영향도 ───────────────────────────────────
with tab2:
    st.subheader("레이어별 불량 영향도 (SHAP)")

    shap_path = "output/layer_shap_importance.csv"
    if os.path.exists(shap_path):
        shap_df = pd.read_csv(shap_path).sort_values("shap_importance", ascending=True)
        fig2 = px.bar(
            shap_df, x="shap_importance", y="layer", orientation="h",
            color="shap_importance", color_continuous_scale="Reds",
            title="레이어별 SHAP 불량 영향도 (높을수록 불량에 큰 영향)",
            labels={"shap_importance": "Mean |SHAP|", "layer": "레이어"},
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("※ SHAP 재계산은 학습(src/train.py) 후 자동 업데이트됩니다.")
    else:
        st.info("SHAP 파일 없음. 먼저 `python src/train.py`를 실행하세요.")

    # 시간 구간별 불량률 (레이어 영향 프록시)
    st.subheader("시간 구간별 불량률 추이")
    defect_trend = df.set_index("timestamp").resample(freq)["label"].mean().dropna()
    fig3 = px.line(
        x=defect_trend.index, y=defect_trend.values,
        labels={"x": "시간", "y": "불량률"},
        title="시간별 불량률 추이",
    )
    fig3.add_hline(y=df["label"].mean(), line_dash="dash", line_color="gray",
                   annotation_text="전체 평균")
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 3: 다음 Glass 예측 ───────────────────────────────
with tab3:
    st.subheader("다음 Glass 불량 확률 예측")

    # 최근 추세 기반 단순 예측 (Logistic on rolling mean)
    roll_mean = df[LAYER_COLS].rolling(30).mean().dropna()
    recent    = roll_mean.tail(n_future)

    total_dpu  = recent.sum(axis=1)
    max_dpu    = recent.max(axis=1)
    nonzero    = (recent > 0).sum(axis=1)
    logit      = -4.5 + 0.02 * total_dpu + 0.01 * max_dpu + 0.3 * nonzero
    pred_probs = 1 / (1 + np.exp(-logit))

    pred_df = pd.DataFrame({
        "glass_index": range(len(pred_probs)),
        "defect_prob": pred_probs.values,
    })

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=pred_df["glass_index"], y=pred_df["defect_prob"],
        fill="tozeroy", name="불량 확률",
        line=dict(color="tomato"),
    ))
    fig4.add_hline(y=0.5, line_dash="dash", line_color="black",
                   annotation_text="판정 임계값 0.5")
    fig4.update_layout(
        title=f"최근 {n_future}개 Glass 불량 예측 확률",
        xaxis_title="Glass 순서",
        yaxis_title="불량 확률",
        yaxis=dict(range=[0, 1]),
        height=400,
    )
    st.plotly_chart(fig4, use_container_width=True)

    high_risk = pred_df[pred_df["defect_prob"] >= 0.5]
    st.metric("예측 불량 Glass 수", f"{len(high_risk)}개 / {len(pred_df)}개",
              delta=f"{len(high_risk)/len(pred_df):.1%}", delta_color="inverse")

# ── Tab 4: 드리프트 모니터링 ─────────────────────────────
with tab4:
    st.subheader("🚨 DPU 드리프트 감지")

    for _, row in trend_df.iterrows():
        layer  = row["layer"]
        slope  = row["slope"]
        status = row["status"]
        is_warn = slope > DRIFT_SLOPE_THRESHOLD

        with st.expander(f"{status}  {layer}  (slope={slope:.4f})", expanded=is_warn):
            y_layer = resampled[layer].values
            x_num   = np.arange(len(y_layer))
            fig_l   = px.line(
                x=resampled.index, y=y_layer,
                labels={"x": "시간", "y": "평균 DPU"},
                title=f"{layer} DPU 추세",
                color_discrete_sequence=["red" if is_warn else "steelblue"],
            )
            trend_line = slope * x_num + y_layer[0]
            fig_l.add_trace(go.Scatter(
                x=resampled.index, y=trend_line,
                mode="lines", line=dict(dash="dash", color="gray"),
                name="추세선",
            ))
            st.plotly_chart(fig_l, use_container_width=True)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("기울기 (slope)", f"{slope:.4f}")
            col_b.metric("R²", f"{row['r_squared']:.4f}")
            col_c.metric("p-value", f"{row['p_value']:.4f}")

            if is_warn:
                st.error(f"⚠️ {layer} DPU 증가 추세 감지 — 공정 점검 권고")

    # 모델 성능 추적
    st.divider()
    st.subheader("모델 성능 추적")
    meta_path = "models/model_latest_meta.json"
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        col1, col2, col3 = st.columns(3)
        col1.metric("학습 시각", meta.get("trained_at", "-"))
        col2.metric("학습 AUC", meta.get("auc", "-"))
        col3.metric("학습 샘플 수", f"{meta.get('n_samples', 0):,}")
    else:
        st.info("모델 메타데이터 없음. `python src/train.py`를 먼저 실행하세요.")
