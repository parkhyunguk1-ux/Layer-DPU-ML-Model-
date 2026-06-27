"""
시계열 기반 모델 학습
- TimeSeriesSplit CV (미래 데이터 누수 방지)
- SHAP으로 레이어별 영향도 추적
- 모델 + 메타데이터 저장
"""

import json
import os
import sys
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.feature_engineering import build_all_features, get_feature_cols

warnings.filterwarnings("ignore")

CSV_PATH   = "data.csv"
TARGET_COL = "label"
MODEL_DIR  = "models"
OUTPUT_DIR = "output"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PARAMS = {
    "max_iter":            500,
    "learning_rate":       0.05,
    "max_leaf_nodes":      63,
    "min_samples_leaf":    50,
    "l2_regularization":   1.0,
    "class_weight":        "balanced",
    "early_stopping":      True,
    "n_iter_no_change":    30,
    "validation_fraction": 0.1,
    "random_state":        42,
}


def train(csv_path: str = CSV_PATH):
    print("=" * 55)
    print("  DPU 시계열 불량 예측 모델 학습")
    print("=" * 55)

    df_raw = pd.read_csv(csv_path)
    print(f"원본 데이터: {df_raw.shape[0]:,}행")

    df = build_all_features(df_raw)

    feat_cols = get_feature_cols(df)
    X = df[feat_cols]
    y = df[TARGET_COL]

    print(f"피처 수: {len(feat_cols)}")
    print(f"불량률: {y.mean():.2%}  (불량={y.sum():,} / 정상={(y==0).sum():,})")

    # TimeSeriesSplit — 과거 → 미래 순서로 학습/검증
    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros(len(y))
    models = []

    print("\n[TimeSeriesSplit 5-Fold 학습]")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = HistGradientBoostingClassifier(**MODEL_PARAMS)
        model.fit(X_tr, y_tr)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"  Fold {fold}  AUC={auc:.4f}  n_iter={model.n_iter_}  "
              f"val_period={df['timestamp'].iloc[val_idx[0]].date()} ~ "
              f"{df['timestamp'].iloc[val_idx[-1]].date()}")
        models.append(model)

    total_auc = roc_auc_score(y, oof_preds)
    print(f"\n[전체 OOF AUC] {total_auc:.4f}")

    y_pred = (oof_preds >= 0.5).astype(int)
    print("\n[분류 리포트]")
    print(classification_report(y, y_pred, target_names=["정상", "불량"]))

    # 최신 데이터로 학습한 마지막 fold 모델 저장 (운영 배포용)
    best_model = models[-1]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{MODEL_DIR}/model_{ts}.pkl"
    meta_path  = f"{MODEL_DIR}/model_{ts}_meta.json"

    joblib.dump(best_model, model_path)

    # 최신 모델 심볼릭 경로
    joblib.dump(best_model, f"{MODEL_DIR}/model_latest.pkl")

    # 메타데이터 저장
    meta = {
        "trained_at":  ts,
        "auc":         round(total_auc, 4),
        "n_features":  len(feat_cols),
        "feature_cols": feat_cols,
        "n_samples":   len(df),
        "defect_rate": round(float(y.mean()), 4),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(f"{MODEL_DIR}/model_latest_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n모델 저장: {model_path}")

    # SHAP 레이어 영향도 분석
    _compute_shap_layer_importance(best_model, X, feat_cols, df)

    return best_model, feat_cols, total_auc


def _compute_shap_layer_importance(model, X, feat_cols, df):
    """레이어별 SHAP 영향도를 시간 구간별로 집계 → CSV + 그래프 저장"""
    print("\nSHAP 레이어 영향도 계산 중...")
    sample = X.sample(min(5000, len(X)), random_state=42)

    explainer  = shap.Explainer(model.predict_proba, sample, output_names=["정상", "불량"])
    shap_vals  = explainer(sample)
    shap_defect = np.abs(shap_vals.values[:, :, 1])   # 불량 클래스 SHAP

    layer_cols = ["하판_RGB", "하판_PAC", "하판_CS", "하판_PI", "상판_BM", "상판_OC", "상판_PI"]
    layer_importance = {}
    for layer in layer_cols:
        idxs = [i for i, c in enumerate(feat_cols) if c.startswith(layer)]
        layer_importance[layer] = shap_defect[:, idxs].sum(axis=1).mean()

    imp_df = pd.DataFrame(
        list(layer_importance.items()), columns=["layer", "shap_importance"]
    ).sort_values("shap_importance", ascending=False)
    imp_df.to_csv(f"{OUTPUT_DIR}/layer_shap_importance.csv", index=False)

    print("\n[레이어별 불량 영향도 (SHAP)]")
    print(imp_df.to_string(index=False))

    # 그래프
    fig, ax = plt.subplots(figsize=(8, 5))
    imp_df.sort_values("shap_importance").plot(
        kind="barh", x="layer", y="shap_importance",
        ax=ax, color="tomato", legend=False
    )
    ax.set_title("레이어별 불량 영향도 (SHAP)")
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/layer_shap.png", dpi=150)
    print(f"SHAP 그래프 저장: {OUTPUT_DIR}/layer_shap.png")


if __name__ == "__main__":
    train()
