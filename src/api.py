"""
FastAPI 예측 서빙
POST /predict      : 단일 Glass 불량 확률 예측
POST /predict/batch: 다수 Glass 배치 예측
GET  /health       : 모델 상태 확인
GET  /metrics      : 최신 모니터링 지표
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.feature_engineering import build_all_features, get_feature_cols

MODEL_PATH = "models/model_latest.pkl"
META_PATH  = "models/model_latest_meta.json"

app = FastAPI(
    title="LCD DPU 불량 예측 API",
    description="레이어별 DPU → 불량(1)/정상(0) 예측",
    version="1.0.0",
)

# 모델 로드
model, feature_cols, meta = None, None, {}

@app.on_event("startup")
def load_model():
    global model, feature_cols, meta
    if not os.path.exists(MODEL_PATH):
        print("⚠️  모델 파일 없음. 먼저 src/train.py를 실행하세요.")
        return
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    print(f"✅ 모델 로드 완료 (학습시각: {meta['trained_at']}, AUC: {meta['auc']})")


class GlassInput(BaseModel):
    timestamp: str
    glass_id:  str
    하판_RGB:  int
    하판_PAC:  int
    하판_CS:   int
    하판_PI:   int
    상판_BM:   int
    상판_OC:   int
    상판_PI:   int


class PredictResponse(BaseModel):
    glass_id:      str
    defect_prob:   float
    prediction:    int
    verdict:       str


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="모델 미로드 상태")
    return {
        "status":      "ok",
        "trained_at":  meta.get("trained_at"),
        "auc":         meta.get("auc"),
        "n_features":  meta.get("n_features"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(glass: GlassInput):
    if model is None:
        raise HTTPException(status_code=503, detail="모델 미로드 상태")

    df = pd.DataFrame([glass.model_dump()])
    df_feat = build_all_features(df)

    # 학습 시 피처와 정합 맞추기
    missing = [c for c in feature_cols if c not in df_feat.columns]
    for c in missing:
        df_feat[c] = 0
    X = df_feat[feature_cols]

    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)

    return PredictResponse(
        glass_id=glass.glass_id,
        defect_prob=round(prob, 4),
        prediction=pred,
        verdict="불량" if pred == 1 else "정상",
    )


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(glasses: list[GlassInput]):
    if model is None:
        raise HTTPException(status_code=503, detail="모델 미로드 상태")

    df = pd.DataFrame([g.model_dump() for g in glasses])
    df_feat = build_all_features(df)

    missing = [c for c in feature_cols if c not in df_feat.columns]
    for c in missing:
        df_feat[c] = 0
    X = df_feat[feature_cols]

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return [
        PredictResponse(
            glass_id=df_feat["glass_id"].iloc[i] if "glass_id" in df_feat.columns else str(i),
            defect_prob=round(float(probs[i]), 4),
            prediction=int(preds[i]),
            verdict="불량" if preds[i] == 1 else "정상",
        )
        for i in range(len(probs))
    ]


@app.get("/metrics")
def metrics():
    trend_path = "output/trend_report.csv"
    shap_path  = "output/layer_shap_importance.csv"
    result = {"model": meta}

    if os.path.exists(trend_path):
        result["dpu_trend"] = pd.read_csv(trend_path).to_dict(orient="records")
    if os.path.exists(shap_path):
        result["layer_shap"] = pd.read_csv(shap_path).to_dict(orient="records")

    return result
