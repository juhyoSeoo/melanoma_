# melanoma_
# CNN 기반 피부암(Melanoma) 이미지 분류 모델 개발 및 성능 분석

##  프로젝트 개요
피부암(흑색종, melanoma) 조기 진단을 통한 생존율 향상을 목적으로, CNN과 VGG16 모델을 활용한 이미지 분류 시스템을 개발하였습니다.

##  프로젝트 목표
- CNN 기반 피부암 이미지 분류 모델 구현
- VGG16 전이학습(Transfer Learning) 모델과의 성능 비교
- 의료 분야 특성을 고려한 Recall 우선 평가
- 실제 임상 환경에서 활용 가능한 모델 개발

##  데이터셋
- **출처**: [Kaggle - Melanoma Skin Cancer Dataset of 10000 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)
- **구성**: 
  - Train set: 10,000장
  - Test set: 1,000장
  - 클래스: 악성(Malignant) vs 양성(Benign)
  - 클래스 균형: 비교적 균등한 분포

##  모델 구조

### 1. CNN (Convolutional Neural Network)
- End-to-End 학습 방식
- ReLU 활성화 함수 사용
- Early Stopping 적용 (Epoch 6에서 과적합 방지)

### 2. VGG16 Feature Extractor
- 사전 훈련된 VGG16 모델의 특성 추출 부분 고정
- 분류층만 새로 학습
- ImageNet 가중치 활용

### 3. VGG16 Fine-tuning
- 전체 네트워크 미세 조정
- Learning rate: 0.0001
- 사전 훈련 가중치 기반 점진적 학습

## 실험 결과

| 모델 | Accuracy | Recall | AUC | 특징 |
|------|----------|--------|-----|------|
| CNN | 90.6% | 87.5% | 0.96 | 높은 재현율, 과적합 경향 |
| VGG16 Feature Extractor | 85.3% | 83.2% | 0.93 | 상대적으로 낮은 성능 |
| VGG16 Fine-tuning | 90.8% | 85.0% | 0.97 | 최고 정확도, 균형잡힌 성능 |

## 주요 분석 결과

### 성능 분석
- **최고 성능**: VGG16 Fine-tuning (AUC: 0.97)
- **Recall 우선**: CNN 모델 (87.5%)
- **False Positive 최소화**: VGG16 Fine-tuning (18건)

### 모델별 특징
1. **CNN**: 의료 이미지에 특화된 높은 재현율
2. **VGG16 Feature Extractor**: 도메인 차이로 인한 제한적 성능
3. **VGG16 Fine-tuning**: 정밀도와 재현율의 균형

## 🛠 기술 스택
- **Framework**: TensorFlow/Keras
- **언어**: Python
- **평가 도구**: 
  - Confusion Matrix
  - ROC Curve
  - AUC Score
- **데이터 전처리**: 
  - 이미지 정규화
  - 데이터 증강 (회전, 확대, 뒤집기)

##  프로젝트 구조
```
melanoma-classification/
├── data/
│   ├── train/
│   └── test/
├── models/
│   ├── cnn_model.py
│   ├── vgg16_feature_extractor.py
│   └── vgg16_finetuning.py
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
├── results/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── performance_metrics.csv
└── README.md
```

##  실행 방법

### 1. 환경 설정
```bash
pip install tensorflow>=2.10.0
pip install keras>=2.10.0
pip install matplotlib seaborn scikit-learn
pip install numpy pandas pillow
```

### 2. 데이터셋 준비
```bash
# Kaggle API 설치 (선택사항)
pip install kaggle

# 데이터셋 다운로드 
kaggle datasets download -d hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images

# 또는 수동으로 다운로드 후 압축 해제
# 링크: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
```

### 3. 프로젝트 구조 설정
```
melanoma-classification/
├── data/
│   ├── train/
│   │   ├── benign/
│   │   └── malignant/
│   └── test/
│       ├── benign/
│       └── malignant/
├── src/
└── results/
```

### 4. 모델 학습 및 평가
```bash
# 데이터 전처리
python src/data_preprocessing.py

# 모델 학습
python src/train_cnn.py
python src/train_vgg16_feature_extractor.py
python src/train_vgg16_finetuning.py

# 평가 및 시각화
python src/evaluate_models.py
```

##  시각화 결과
- **Learning Curves**: 각 모델의 학습 과정
- **Confusion Matrix**: 분류 성능 상세 분석
- **ROC Curves**: 전체 성능 비교

##  임상적 의의
- **스크리닝 단계**: CNN 모델 (높은 재현율)
- **정밀 진단**: VGG16 Fine-tuning (균형잡힌 성능)
- **오진 최소화**: False Positive 감소를 통한 불필요한 추가 검사 방지

##  향후 계획
- [ ] 다중 분류 모델 확장 (여러 피부암 유형)
- [ ] 앙상블 기법 적용
- [ ] Focal Loss를 통한 재현율 개선
- [ ] ResNet, EfficientNet 등 최신 아키텍처 비교
- [ ] 더 큰 규모의 데이터셋 활용

##  한계점
- 20,000장의 제한된 데이터셋
- 이진 분류에 국한
- CNN 모델의 과적합 이슈

##  팀원
- **서주효** (202201743)
- **신재연** (202302009)

##  참고문헌
- [Kaggle - Melanoma Skin Cancer Dataset](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)
- IBM Developer Artificial Intelligence
- GeeksforGeeks VGG-16 CNN Model

##  라이선스
MIT License

---
*이 프로젝트는 데이터마이닝 수업의 일환으로 진행되었으며, 의료 이미지 분류에서의 딥러닝 모델 비교 연구입니다.*
