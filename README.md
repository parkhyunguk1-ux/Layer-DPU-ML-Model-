# 🔬 LCD Layer DPU MLOps

> LCD 임베딩 공정에서 **레이어별 DPU(Defects Per Unit) 시계열 추세**를 분석하고  
> **다음 Glass의 불량 여부를 실시간 예측**하는 풀스택 MLOps 시스템입니다.

---

## 🎯 시스템 목표

| 목표 | 설명 |
|------|------|
| 📈 DPU 증가 추세 감지 | 시간에 따라 어느 레이어의 DPU가 증가하는지 자동 탐지 |
| 🎯 레이어 영향도 추적 | SHAP으로 어떤 레이어가 불량에 가장 크게 기여하는지 시간 흐름별 추적 |
| 🔮 다음 Glass 불량 예측 | 직전 N개 Glass의 DPU 추세 → 다음 Glass 불량 확률 예측 |
| 🚨 드리프트 모니터링 | 모델 성능 열화 및 DPU 분포 변화 자동 감지 → 재학습 권고 |

---

## 🗂 프로젝트 구조

```
Layer-DPU-ML-Model/
├── pipeline.py                   # MLOps 파이프라인 엔트리포인트
├── generate_sample_data.py       # 시계열 샘플 데이터 생성 (10만 건)
├── requirements.txt
│
├── src/
│   ├── feature_engineering.py   # 시계열 피처 생성 (rolling, slope, lag)
│   ├── train.py                 # TimeSeriesSplit 학습 + SHAP 분석
│   ├── monitor.py               # DPU 드리프트 감지 + 성능 모니터링
│   └── api.py                   # FastAPI 예측 서빙
│
├── dashboard/
│   └── app.py                   # Streamlit 대시보드
│
├── models/
│   ├── model_latest.pkl         # 운영 배포 모델
│   └── model_latest_meta.json   # 모델 메타데이터 (AUC, 피처 목록 등)
│
└── output/
    ├── dpu_trend.png            # 레이어별 DPU 추세 그래프
    ├── layer_shap.png           # SHAP 레이어 영향도 그래프
    ├── trend_report.csv         # 레이어별 slope / p-value 리포트
    └── layer_shap_importance.csv
```

---

## ⚙️ 설치

```bash
pip install -r requirements.txt
```

---

## 🚀 실행 방법

### 전체 파이프라인 한 번에 실행

```bash
python pipeline.py --step all
```

### 단계별 실행

```bash
# 1. 샘플 데이터 생성 (실제 데이터 있으면 생략)
python pipeline.py --step data

# 2. DPU 드리프트 모니터링
python pipeline.py --step monitor

# 3. 모델 학습
python pipeline.py --step train

# 4. API 서버 실행 (http://localhost:8000/docs)
python pipeline.py --step api

# 5. 대시보드 실행 (http://localhost:8501)
python pipeline.py --step dashboard
```

---

## 🧠 MLOps 아키텍처

```
CSV (timestamp + glass_id + 레이어별 DPU + label)
        │
        ▼
┌─────────────────────────────┐
│   Feature Engineering        │
│  - rolling mean/std (10/30/100 window)  │
│  - slope (추세 방향)          │
│  - trend_signal (단기-장기)   │
│  - lag 피처 (직전 1/3/5 Glass)│
│  - 하판/상판 소계, large_dpu  │
└────────────┬────────────────┘
             │
        ┌────▼────┐
        │  Train   │  TimeSeriesSplit 5-Fold
        │          │  (미래 데이터 누수 방지)
        └────┬─────┘
             │
     ┌───────┼───────┐
     ▼       ▼       ▼
  Monitor   SHAP    API
  드리프트  레이어  FastAPI
  감지     영향도   서빙
     │       │       │
     └───────┴───────┘
                │
           Dashboard
         Streamlit 시각화
```

---

## 📊 피처 엔지니어링

### 시계열 피처

| 피처 | 설명 |
|------|------|
| `{layer}_roll10_mean` | 직전 10 Glass 평균 DPU |
| `{layer}_roll30_slope` | 직전 30 Glass DPU 기울기 (양수 = 증가) |
| `{layer}_trend_signal` | 단기(10) - 장기(100) 평균 차이 (급등 감지) |
| `{layer}_lag1` | 직전 1 Glass DPU |
| `{layer}_lag3` | 직전 3 Glass DPU |

### 집계 피처

| 피처 | 설명 |
|------|------|
| `total_dpu` | 전체 레이어 DPU 합 |
| `large_dpu_count` | DPU ≥ 50인 레이어 수 |
| `하판_total` / `상판_total` | 판별 소계 |

---

## 📈 대시보드 구성

| 탭 | 내용 |
|----|------|
| 📈 DPU 추세 분석 | 레이어별 시계열 + 추세선 (빨간색 = 증가 경고) |
| 🎯 레이어 영향도 | SHAP 기반 불량 기여도 바 차트 + 불량률 추이 |
| 🔮 다음 Glass 예측 | 향후 N개 Glass 불량 확률 예측 그래프 |
| 🚨 드리프트 모니터링 | 레이어별 slope/p-value + 모델 성능 추적 |

---

## 🌐 API 엔드포인트

```
POST /predict        단일 Glass 불량 예측
POST /predict/batch  다수 Glass 배치 예측
GET  /health         모델 상태 확인
GET  /metrics        DPU 추세 + SHAP 지표 조회
```

### 예측 요청 예시

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-06-01T10:00:00",
    "glass_id":  "GL000001",
    "하판_RGB": 0,
    "하판_PAC": 15,
    "하판_CS":  0,
    "하판_PI":  80,
    "상판_BM":  50,
    "상판_OC":  0,
    "상판_PI":  0
  }'
```

### 응답

```json
{
  "glass_id":    "GL000001",
  "defect_prob": 0.7823,
  "prediction":  1,
  "verdict":     "불량"
}
```

---

## 🚨 드리프트 감지 기준

| 항목 | 기준 | 조치 |
|------|------|------|
| DPU slope > 0.05 | 레이어 DPU 증가 추세 | 공정 점검 권고 |
| AUC 하락 > 0.03 | 모델 성능 열화 | 재학습 권고 |

---

## 📁 CSV 입력 형식

| 컬럼 | 설명 | 예시 |
|------|------|------|
| `timestamp` | Glass 생산 시각 | `2024-01-01 09:00:00` |
| `glass_id` | Glass 고유 ID | `GL000001` |
| `하판_RGB` | 하판 RGB 레이어 DPU | 0, 15, 30, 50, 80, 100 |
| `하판_PAC` | 하판 PAC 레이어 DPU | 0, 15, 30, 50, 80 |
| `하판_CS` | 하판 CS 레이어 DPU | 0, 30, 50 |
| `하판_PI` | 하판 PI 레이어 DPU | 0, 15, 80 |
| `상판_BM` | 상판 BM 레이어 DPU | 0, 30, 50 |
| `상판_OC` | 상판 OC 레이어 DPU | 0, 15, 50 |
| `상판_PI` | 상판 PI 레이어 DPU | 0, 80, 100 |
| **`label`** | **불량(1) / 정상(0)** | **0 or 1** |

---

## 🔄 자동 재학습 스케줄 (운영 환경)

```bash
# cron 예시: 매일 새벽 2시 재학습
0 2 * * * cd /path/to/project && python pipeline.py --step train
```

---

## 📦 환경

- Python 3.9+
- scikit-learn, scipy, shap
- FastAPI + uvicorn
- Streamlit + Plotly
