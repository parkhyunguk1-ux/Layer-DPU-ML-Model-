"""
저장된 모델로 새 데이터 불량 예측
"""

import pandas as pd
import joblib
import sys

MODEL_PATH = "output/lgbm_model.pkl"
TARGET_COL = "label"


def predict(csv_path: str):
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(csv_path)

    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    # 파생 피처 (train.py와 동일하게)
    feature_cols = [c for c in df.columns]
    df["total_dpu"]      = df[feature_cols].sum(axis=1)
    df["max_dpu"]        = df[feature_cols].max(axis=1)
    df["mean_dpu"]       = df[feature_cols].mean(axis=1)
    df["std_dpu"]        = df[feature_cols].std(axis=1)
    df["nonzero_layers"] = (df[feature_cols] > 0).sum(axis=1)

    proba = model.predict_proba(df)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    result = pd.DataFrame({
        "불량확률": proba.round(4),
        "예측결과": pred,
        "판정": ["불량" if p == 1 else "정상" for p in pred],
    })
    out_path = csv_path.replace(".csv", "_predicted.csv")
    result.to_csv(out_path, index=False)
    print(f"예측 완료 → {out_path}")
    print(result.head(10))


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "new_data.csv"
    predict(csv_path)
