# 🔬 Layer DPU ML Model

> LCD 임베딩 공정에서 **레이어별 DPU(Defects Per Unit)** 를 입력으로 받아  
> 최종 제품의 **불량(1) / 정상(0)** 을 예측하는 머신러닝 분류 모델입니다.

---

## 📌 배경

LCD 제조 공정에서 각 레이어마다 DPU가 측정됩니다.  
이 값들을 종합하면 최종 불량 여부를 **사전에 예측**할 수 있어, Repair 우선순위 결정과 수율 개선에 활용됩니다.

| 판 | 레이어 | 설명 |
|----|--------|------|
| 하판 | RGB | Red / Green / Blue 픽셀 레이어 |
| 하판 | PAC | Photo Alignment Coating |
| 하판 | CS | Color Shield |
| 하판 | PI | Polyimide (배향막) |
| 상판 | BM | Black Matrix |
| 상판 | OC | Over Coating |
| 상판 | PI | Polyimide (배향막) |

**DPU 구간:** `0` (정상) · `15` · `30` · `50` · `80` · `100` (80이상)

---

## 🗂 프로젝트 구조

```
Layer-DPU-ML-Model/
├── train.py                  # 모델 학습 (5-Fold CV + HistGradientBoosting)
├── predict.py                # 저장된 모델로 새 데이터 예측
├── generate_sample_data.py   # 테스트용 샘플 데이터 생성 (10만 건)
├── requirements.txt          # 패키지 의존성
└── output/
    ├── model.pkl             # 학습된 모델
    ├── feature_importance.csv
    └── results.png           # 혼동 행렬 + 피처 중요도 그래프
```

---

## ⚙️ 설치

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
joblib>=1.3
```

---

## 🚀 사용법

### 1. 샘플 데이터 생성 (실제 데이터가 없을 때)

```bash
python generate_sample_data.py
# → data.csv 생성 (100,000행)
```

### 2. 모델 학습

`train.py` 상단 설정을 실제 환경에 맞게 수정합니다.

```python
CSV_PATH   = "data.csv"   # 실제 CSV 경로
TARGET_COL = "label"      # 불량(1)/정상(0) 컬럼명

LAYER_COLS = [
    "하판_RGB", "하판_PAC", "하판_CS", "하판_PI",
    "상판_BM",  "상판_OC",  "상판_PI",
]
```

```bash
python train.py
```

**실행 예시 출력:**
```
=======================================================
  DPU 기반 LCD 불량 예측 모델  (HistGradientBoosting)
=======================================================
데이터 로드: 100,000행 × 8열
불량률: 34.32%  (불량=34,319 / 정상=65,681)

피처 수: 15

[5-Fold 교차검증]
  Fold 1  AUC=0.8553  n_iter=110
  Fold 2  AUC=0.8569  n_iter=116
  Fold 3  AUC=0.8589  n_iter=111
  Fold 4  AUC=0.8545  n_iter=106
  Fold 5  AUC=0.8597  n_iter=108

[전체 OOF AUC] 0.8570
```

### 3. 새 데이터 예측

```bash
python predict.py new_data.csv
# → new_data_predicted.csv 생성
```

**출력 컬럼:**

| 컬럼 | 설명 |
|------|------|
| `불량확률` | 0 ~ 1 사이 불량 예측 확률 |
| `예측결과` | 0 또는 1 |
| `판정` | 정상 / 불량 |

---

## 🧠 모델 구조

```
입력: 레이어별 DPU (7개 컬럼)
        ↓
피처 엔지니어링
  ├── total_dpu       : 전체 DPU 합계       ← 중요도 1위
  ├── max_dpu         : 최대 DPU
  ├── mean_dpu        : 평균 DPU
  ├── std_dpu         : DPU 표준편차
  ├── nonzero_layers  : DPU > 0 레이어 수
  ├── 하판_total      : 하판 레이어 DPU 합
  ├── 상판_total      : 상판 레이어 DPU 합
  └── large_dpu_count : DPU ≥ 50 레이어 수
        ↓
HistGradientBoostingClassifier
  - 5-Fold Stratified Cross Validation
  - Early Stopping (30 rounds)
  - class_weight = balanced  (불균형 데이터 대응)
        ↓
출력: 불량 확률 + 정상/불량 판정
```

---

## 📊 성능 결과

| 지표 | 정상(0) | 불량(1) |
|------|---------|---------|
| Precision | 0.86 | 0.64 |
| Recall | 0.78 | 0.77 |
| F1-score | 0.82 | 0.70 |
| **OOF AUC** | **0.857** | |

> 샘플 데이터 기준 결과입니다. 실제 공정 데이터로 학습 시 성능이 달라질 수 있습니다.

---

## 🔧 파라미터 튜닝 가이드

| 파라미터 | 기본값 | 높이면 | 낮추면 |
|----------|--------|--------|--------|
| `max_iter` | 500 | 더 오래 학습 | 빠르게 종료 |
| `learning_rate` | 0.05 | 수렴 빠름 (과적합 위험) | 더 안정적 |
| `max_leaf_nodes` | 63 | 복잡한 패턴 학습 | 과적합 방지 |
| `min_samples_leaf` | 50 | 과적합 방지 | 세밀한 분할 |

---

## 🔄 LightGBM / XGBoost 업그레이드

현재는 `scikit-learn`의 `HistGradientBoostingClassifier`를 사용합니다.  
더 높은 성능(AUC 0.90+)을 원하면 아래와 같이 업그레이드하세요.

```bash
# Mac
brew install libomp
pip install lightgbm

# train.py 상단 변경
import lightgbm as lgb
model = lgb.LGBMClassifier(...)
```

---

## 📁 CSV 데이터 형식

| 컬럼명 | 설명 | 값 |
|--------|------|----|
| `하판_RGB` | 하판 RGB 레이어 DPU | 0, 15, 30, 50, 80, 100 |
| `하판_PAC` | 하판 PAC 레이어 DPU | 0, 15, 30, 50, 80 |
| `하판_CS` | 하판 CS 레이어 DPU | 0, 30, 50 |
| `하판_PI` | 하판 PI 레이어 DPU | 0, 15, 80 |
| `상판_BM` | 상판 BM 레이어 DPU | 0, 30, 50 |
| `상판_OC` | 상판 OC 레이어 DPU | 0, 15, 50 |
| `상판_PI` | 상판 PI 레이어 DPU | 0, 80, 100 |
| **`label`** | **불량(1) / 정상(0)** | **0 or 1** |

---

## 📦 환경

- Python 3.8+
- scikit-learn >= 1.3
- pandas >= 2.0
- numpy >= 1.24
- matplotlib >= 3.7
- joblib >= 1.3
