"""
레이어별 DPU → 불량(1)/정상(0) 분류 모델
GradientBoosting 기반 파이프라인 (sklearn)

레이어 구성:
  하판: RGB, PAC, CS, PI
  상판: BM, OC, PI
DPU 구간: 0(정상), 15, 30, 50, 80, 80이상(100)

※ LightGBM/XGBoost로 교체하려면:
    brew install libomp  # Mac
    pip install lightgbm  or  xgboost
   그 후 train_lgbm.py 사용
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ── 설정 ──────────────────────────────────────────────────────────────
CSV_PATH   = "data.csv"   # 실제 CSV 경로로 변경
TARGET_COL = "label"      # 불량(1)/정상(0) 컬럼명
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 레이어별 DPU 컬럼 (실제 CSV 컬럼명이 다르면 여기만 수정)
LAYER_COLS = [
    "하판_RGB",
    "하판_PAC",
    "하판_CS",
    "하판_PI",
    "상판_BM",
    "상판_OC",
    "상판_PI",
]

# HistGradientBoosting: sklearn의 가장 빠른 GBDT (LightGBM과 유사)
MODEL_PARAMS = {
    "max_iter":         500,
    "learning_rate":    0.05,
    "max_leaf_nodes":   63,
    "max_depth":        None,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "random_state":     42,
    "class_weight":     "balanced",   # 불균형 대응
    "early_stopping":   True,
    "n_iter_no_change": 30,
    "validation_fraction": 0.1,
}
# ─────────────────────────────────────────────────────────────────────


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"데이터 로드: {df.shape[0]:,}행 × {df.shape[1]}열")
    n_defect = int(df[TARGET_COL].sum())
    n_normal = int((df[TARGET_COL] == 0).sum())
    print(f"불량률: {df[TARGET_COL].mean():.2%}  (불량={n_defect:,} / 정상={n_normal:,})")
    return df


def build_features(df: pd.DataFrame):
    """레이어 DPU + 파생 피처 생성"""
    X = df[LAYER_COLS].copy()
    y = df[TARGET_COL].copy()

    # 파생 피처: 레이어 간 관계
    X["total_dpu"]      = X[LAYER_COLS].sum(axis=1)
    X["max_dpu"]        = X[LAYER_COLS].max(axis=1)
    X["mean_dpu"]       = X[LAYER_COLS].mean(axis=1)
    X["std_dpu"]        = X[LAYER_COLS].std(axis=1)
    X["nonzero_layers"] = (X[LAYER_COLS] > 0).sum(axis=1)

    # 하판 / 상판 소계
    lower_cols = ["하판_RGB", "하판_PAC", "하판_CS", "하판_PI"]
    upper_cols = ["상판_BM", "상판_OC", "상판_PI"]
    X["하판_total"]      = X[lower_cols].sum(axis=1)
    X["상판_total"]      = X[upper_cols].sum(axis=1)

    # 대형 결함 레이어 수 (DPU ≥ 50)
    X["large_dpu_count"] = (X[LAYER_COLS] >= 50).sum(axis=1)

    print(f"\n피처 수: {X.shape[1]}")
    print(f"피처 목록: {list(X.columns)}")
    return X, y


def cross_validate(X: pd.DataFrame, y: pd.Series):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    models = []

    print("\n[5-Fold 교차검증 — HistGradientBoostingClassifier]")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = HistGradientBoostingClassifier(**MODEL_PARAMS)
        model.fit(X_tr, y_tr)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"  Fold {fold}  AUC={auc:.4f}  n_iter={model.n_iter_}")
        models.append(model)

    total_auc = roc_auc_score(y, oof_preds)
    print(f"\n[전체 OOF AUC] {total_auc:.4f}")
    return models, oof_preds


def compute_importance(models: list, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Permutation importance (5-Fold 평균) — HistGBT는 feature_importances_ 미지원"""
    perm_list = []
    for model in models:
        r = permutation_importance(
            model, X, y, n_repeats=5, random_state=42,
            scoring="roc_auc", n_jobs=-1,
        )
        perm_list.append(r.importances_mean)
    importance = np.mean(perm_list, axis=0)
    return pd.Series(importance, index=X.columns)


def plot_results(y: pd.Series, oof_preds: np.ndarray, models: list, X: pd.DataFrame):
    y_pred = (oof_preds >= 0.5).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DPU 기반 불량 예측 모델 결과", fontsize=14)

    # 혼동 행렬
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["정상(0)", "불량(1)"])
    disp.plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion Matrix")

    # Permutation 피처 중요도
    print("\n피처 중요도 계산 중 (permutation importance)...")
    feat_imp = compute_importance(models, X, y).sort_values(ascending=True)
    feat_imp.tail(15).plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_title("Permutation Importance (Top 15, 5-Fold avg)")
    axes[1].set_xlabel("Mean AUC decrease")

    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n결과 이미지 저장: {save_path}")

    print("\n[분류 리포트]")
    print(classification_report(y, y_pred, target_names=["정상", "불량"]))


def save_feature_importance(models: list, X: pd.DataFrame, y: pd.Series):
    feat_imp_series = compute_importance(models, X, y)
    feat_imp = pd.DataFrame({
        "feature":    feat_imp_series.index,
        "importance": feat_imp_series.values,
    }).sort_values("importance", ascending=False)
    path = f"{OUTPUT_DIR}/feature_importance.csv"
    feat_imp.to_csv(path, index=False)
    print(f"피처 중요도 저장: {path}")
    print(feat_imp.to_string(index=False))


def main():
    print("=" * 55)
    print("  DPU 기반 LCD 불량 예측 모델  (HistGradientBoosting)")
    print("=" * 55)

    df = load_data(CSV_PATH)
    X, y = build_features(df)
    models, oof_preds = cross_validate(X, y)

    plot_results(y, oof_preds, models, X)
    save_feature_importance(models, X, y)

    # 최고 AUC fold 모델 저장
    fold_aucs = [roc_auc_score(y, m.predict_proba(X)[:, 1]) for m in models]
    best_idx  = int(np.argmax(fold_aucs))
    model_path = f"{OUTPUT_DIR}/model.pkl"
    joblib.dump(models[best_idx], model_path)
    print(f"\n최적 모델 저장 (Fold {best_idx + 1}): {model_path}")


if __name__ == "__main__":
    main()
