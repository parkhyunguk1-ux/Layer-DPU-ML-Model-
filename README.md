# DPU 기반 LCD 불량 예측 모델

레이어별 DPU(Defects Per Unit)를 입력으로 받아 최종 불량(1) / 정상(0)을 예측하는 LightGBM 분류 모델입니다.

## 배경

LCD 임베딩 공정에서 각 레이어(하판: RGB·PAC·CS·PI / 상판: BM·OC·PI)마다 DPU가 측정되며,  
이 값들을 종합해 제품 최종 불량 여부를 사전 예측합니다.

## 프로젝트 구조

```
dpu-defect-model/
├── train.py          # 모델 학습 (5-Fold CV + LightGBM)
├── predict.py        # 저장된 모델로 새 데이터 예측
├── requirements.txt  # 패키지 의존성
└── output/           # 학습 후 자동 생성
    ├── lgbm_model.pkl
    └── results.png
```

## 사용법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. `train.py` 상단 설정 수정

```python
CSV_PATH   = "data.csv"   # CSV 파일 경로
TARGET_COL = "label"      # 불량(1)/정상(0) 컬럼명
```

### 3. 학습 실행

```bash
python train.py
```

**출력 예시:**
```
데이터 로드: 100,000행 × 8열
불량률: 3.21%  (불량=3,210 / 정상=96,790)
피처 수: 7  |  컬럼: [하판_RGB, 하판_PAC, 하판_CS, 하판_PI, 상판_BM, 상판_OC, 상판_PI]

[5-Fold 교차검증 시작]
  Fold 1  AUC=0.9712  best_iter=342
  Fold 2  AUC=0.9698  best_iter=387
  ...
[전체 OOF AUC] 0.9705
```

### 4. 새 데이터 예측

```bash
python predict.py new_data.csv
# → new_data_predicted.csv 생성
```

## 입력 데이터 형식

| 컬럼명 | 설명 | 값 예시 |
|--------|------|---------|
| 하판_RGB | 하판 Red/Green/Blue DPU | 0, 15, 30, 50, 80, 100 |
| 하판_PAC | 하판 PAC 레이어 DPU | 0, 15, 30, 50, 80 |
| 하판_CS  | 하판 CS 레이어 DPU | 0, 30, 50 |
| 하판_PI  | 하판 PI 레이어 DPU | 0, 15, 80 |
| 상판_BM  | 상판 BM 레이어 DPU | 0, 30, 50 |
| 상판_OC  | 상판 OC 레이어 DPU | 0, 15, 50 |
| 상판_PI  | 상판 PI 레이어 DPU | 0, 80, 100 |
| **label** | **불량(1) / 정상(0)** | **0 or 1** |

## 모델 구조

```
입력 피처 (레이어별 DPU 7개)
  ↓
파생 피처 생성
  - total_dpu      : 전체 DPU 합계
  - max_dpu        : 최대 DPU
  - mean_dpu       : 평균 DPU
  - std_dpu        : DPU 표준편차
  - nonzero_layers : DPU > 0인 레이어 수
  ↓
LightGBM (GBDT, binary classification)
  - 5-Fold Stratified Cross Validation
  - Early Stopping (50 rounds)
  - class_weight=balanced (불균형 데이터 대응)
  ↓
출력: 불량 확률(0~1) + 불량/정상 판정
```

## 평가 지표

- **AUC-ROC**: 불균형 데이터에서 가장 신뢰도 높은 지표
- **Confusion Matrix**: 실제 불량을 얼마나 잡아냈는지 확인
- **Classification Report**: Precision / Recall / F1

## 주요 파라미터 튜닝 가이드

| 파라미터 | 기본값 | 높이면 | 낮추면 |
|----------|--------|--------|--------|
| `num_leaves` | 63 | 더 복잡한 패턴 학습 | 과적합 방지 |
| `learning_rate` | 0.05 | 수렴 빠름 (과적합 위험) | 더 안정적 |
| `min_child_samples` | 50 | 과적합 방지 | 더 세밀한 분할 |
| `n_estimators` | 1000 | Early Stopping으로 자동 조절 | — |

## 요구사항

- Python 3.8+
- lightgbm >= 4.0
- scikit-learn >= 1.3
- pandas >= 2.0
