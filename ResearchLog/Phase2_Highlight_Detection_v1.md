# Phase 2: 하이라이트 감지 및 OCR 모듈 개발

**단계**: Phase 2 (Week 3-4)
**버전**: v1.0
**작성일**: 2025-01-19
**상태**: 🔄 진행 중

---

## 📋 개요

HSV 색공간 기반 하이라이트 영역 감지 시스템과 성능 평가 프레임워크를 구현했습니다. 초기 테스트 결과 **mIoU 0.0**으로 목표치(0.75)에 크게 미달하여 HSV 범위 최적화가 필수적임을 확인했습니다.

---

## 🎯 목표

- [ ] HSV 색공간 기반 하이라이트 감지 (mIoU > 0.75)
- [x] IoU 기반 성능 평가 시스템 구축
- [x] 하이퍼파라미터 최적화 프레임워크
- [ ] Tesseract OCR 텍스트 추출 (CER < 5%)
- [ ] OCR 성능 평가 시스템

---

## 🏗️ 구현 내용

### 1. 하이라이트 감지 모듈 (Week 3)

**파일**: `src/highlight_detector/highlight_detector.py`

#### HighlightDetector 클래스

**핵심 기능**:
- HSV 색공간 변환
- 색상 범위 기반 마스킹
- 모폴로지 연산 (Opening, Closing)
- 컨투어 검출 및 바운딩 박스 추출

**기본 HSV 범위**:
```python
DEFAULT_HSV_RANGES = {
    'yellow': {
        'lower': np.array([20, 100, 100]),
        'upper': np.array([30, 255, 255])
    },
    'green': {
        'lower': np.array([40, 40, 40]),
        'upper': np.array([80, 255, 255])
    },
    'pink': {
        'lower': np.array([140, 50, 50]),
        'upper': np.array([170, 255, 255])
    }
}
```

**알고리즘 파이프라인**:
```python
def detect(self, image):
    # 1. BGR to HSV 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2. 각 색상별 마스크 생성
    mask = cv2.inRange(hsv, lower, upper)

    # 3. 모폴로지 연산
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 구멍 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 노이즈 제거

    # 4. 컨투어 검출
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. 바운딩 박스 추출
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            detections.append({'bbox': [x, y, w, h], 'color': color})
```

**파라미터**:
- `kernel_size`: (5, 5) - 모폴로지 커널 크기
- `min_area`: 100 - 최소 컨투어 면적 (노이즈 필터링)
- `morph_iterations`: 1 - 모폴로지 연산 반복 횟수

---

### 2. 성능 평가 시스템

**파일**: `src/highlight_detector/evaluator.py`

#### HighlightEvaluator 클래스

**평가 지표**:

**1. IoU (Intersection over Union)**
```python
def calculate_iou(bbox1, bbox2):
    intersection_area = calculate_intersection(bbox1, bbox2)
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area
```

**2. mIoU (mean IoU)**
- 매칭된 모든 detection의 IoU 평균
- 목표: > 0.75

**3. Precision, Recall, F1-Score**
```python
TP = len(matches)  # 올바르게 감지한 하이라이트
FP = len(unmatched_predictions)  # 잘못 감지한 것
FN = len(unmatched_ground_truths)  # 놓친 하이라이트

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**매칭 알고리즘**:
- Greedy matching (highest IoU first)
- IoU threshold: 0.5
- 색상 일치 필수 (yellow ↔ yellow만 매칭)

---

### 3. 하이퍼파라미터 최적화 프레임워크

**파일**: `src/highlight_detector/optimizer.py`

#### ParameterOptimizer 클래스

**최적화 대상**:

**1. HSV 색상 범위**
```python
def grid_search_hsv(color, h_range, s_range, v_range, step=5):
    # Hue, Saturation, Value 조합 탐색
    for h_lower, h_upper, s_lower, v_lower in combinations:
        # 각 조합으로 감지 테스트
        # mIoU 계산 및 최적값 업데이트
```

**2. 모폴로지 파라미터**
```python
def grid_search_morph(kernel_sizes, min_areas, iterations):
    # 커널 크기, 최소 영역, 반복 횟수 조합 탐색
```

**탐색 범위**:
- Yellow: H=[15, 35], S=[40, 100], V=[40, 100]
- Green: H=[35, 85], S=[40, 100], V=[40, 100]
- Pink: H=[135, 175], S=[40, 100], V=[40, 100]
- Kernel: (3,3), (5,5), (7,7)
- Min area: 50, 100, 150, 200
- Iterations: 1, 2

---

## 🧪 초기 테스트 결과

### 테스트 설정

**데이터**: Validation 10개 샘플
**실행 명령**: `uv run python test_highlight_detection.py`

### 결과

**전체 성능**:
```
mIoU:      0.0000  ❌ (목표: 0.75)
Precision: 0.0000  ❌
Recall:    0.0000  ❌
F1-Score:  0.0000  ❌
```

**감지 통계**:
```
True Positives:  0
False Positives: 1257
False Negatives: 22
```

**문제점**:
1. **단 하나의 하이라이트도 올바르게 감지하지 못함** (TP=0)
2. **1257개의 오감지** (대부분 노이즈)
3. **모든 실제 하이라이트를 놓침** (FN=22)

### 샘플별 상세 결과

| 이미지 | GT | 감지 | Precision | Recall | mIoU |
|--------|----|----|-----------|--------|------|
| validation_0000_orig | 1 | 0 | 0.000 | 0.000 | 0.000 |
| validation_0001_aug0 | 1 | 0 | 0.000 | 0.000 | 0.000 |
| validation_0002_aug1 | 1 | 0 | 0.000 | 0.000 | 0.000 |
| validation_0003_orig | 4 | 1 | 0.000 | 0.000 | 0.000 |
| validation_0004_aug0 | 4 | **757** | 0.000 | 0.000 | 0.000 |

**특이 사항**:
- 원본(orig) 이미지: 대부분 감지 안됨 (0~1개)
- 증강(aug) 이미지: 과도한 오감지 (757개)
- 증강이 색상 변화를 일으켜 문제 악화

---

## 🐛 발견된 문제 및 분석

### 문제 1: HSV 범위 불일치 ⚠️

**원인 분석**:

합성 데이터 생성 시 사용한 BGR 색상:
```python
# data_generator/highlight_overlay.py
HIGHLIGHT_COLORS = {
    'yellow': (0, 255, 255),   # BGR
    'green': (0, 255, 0),
    'pink': (203, 192, 255)
}
```

투명도 적용 (alpha=0.3):
```python
highlighted = original * 0.7 + color * 0.3
```

**실제 렌더링된 색상**:
- 흰 배경(255, 255, 255) + 노랑(0, 255, 255) @ alpha=0.3
- = (178.5, 255, 255) in BGR
- ≈ HSV에서 Hue는 변하지만 Saturation/Value가 낮아짐

**기본 HSV 범위와의 불일치**:
```python
# 기본값
'yellow': {
    'lower': [20, 100, 100],  # 높은 S, V 요구
    'upper': [30, 255, 255]
}

# 실제 (예상)
# Saturation: ~50-80 (투명도로 인해 낮음)
# Value: ~200-255 (밝음)
```

### 문제 2: 증강으로 인한 색상 왜곡

**증강 효과**:
- `RandomBrightnessContrast`: ±20% 밝기/대비 변화
- `HueSaturationValue`: Hue ±10, Sat ±15, Val ±10
- `ImageCompression`: JPEG 압축 artifacts

**영향**:
- HSV 색공간에서 색상이 더욱 불안정해짐
- 원본도 감지 못하는데 증강은 더 어려움

### 문제 3: 과도한 오감지

**757개 오감지 분석**:
```python
# 감지된 bbox 예시
yellow: [36, 980, 12, 15], conf=0.70  # 이미지 하단 노이즈
yellow: [644, 979, 13, 14], conf=0.62  # 작은 노이즈
```

**원인**:
1. HSV 범위가 너무 넓어서 배경의 미세한 색상 변화도 포착
2. `min_area=100`이 너무 작아서 작은 노이즈도 통과
3. 증강으로 인한 노이즈가 yellow 범위에 걸림

---

## 💡 해결 방안

### 1. HSV 범위 재측정 (우선순위: 높음)

**방법**: 실제 합성 하이라이트의 HSV 값 측정 도구 개발

**계획**:
```python
# analyze_highlight_colors.py
def measure_actual_hsv():
    # 1. 원본 이미지 로드
    # 2. Ground truth 하이라이트 영역만 추출
    # 3. HSV로 변환 후 평균/분산 계산
    # 4. 색상별 실제 범위 도출
```

### 2. 파라미터 최적화 실행

```bash
# optimizer.py 실행
uv run python -m src.highlight_detector.optimizer
```

**예상 조정**:
- Saturation lower: 100 → 40
- Value lower: 100 → 150
- `min_area`: 100 → 200 (노이즈 필터링 강화)

### 3. 원본 이미지 우선 최적화

**전략**:
1. 증강 이미지는 제외하고 원본(orig)만 사용
2. 원본에서 먼저 mIoU > 0.75 달성
3. 그 후 증강 이미지로 일반화 테스트

### 4. 디버깅 도구 개발

**필요 도구**:
- HSV 시각화 도구 (색공간 분포 확인)
- 단계별 이미지 출력 (mask, morphology 등)
- 실패 케이스 자동 분류

---

## 📊 다음 단계 (우선순위)

### 우선순위 1: HSV 범위 분석 및 조정

**작업**:
1. 실제 하이라이트 HSV 측정 스크립트 작성
2. 색상별 분포 분석
3. 최적 범위 도출
4. 수동 조정 테스트

**예상 소요**: 2-4시간

### 우선순위 2: 원본 이미지 최적화

**작업**:
1. 원본(orig) 필터링 테스트 스크립트
2. 파라미터 그리드 서치
3. mIoU > 0.75 목표 달성

**예상 소요**: 4-6시간

### 우선순위 3: 증강 이미지 대응

**작업**:
1. 증강별 색상 변화 분석
2. 강건한 HSV 범위 확장
3. 전체 데이터셋 재평가

**예상 소요**: 2-3시간

---

## 📁 생성된 파일 목록

### 소스 코드
- `src/highlight_detector/__init__.py`
- `src/highlight_detector/highlight_detector.py` (319 lines)
- `src/highlight_detector/evaluator.py` (295 lines)
- `src/highlight_detector/optimizer.py` (288 lines)

### 테스트 스크립트
- `test_highlight_detection.py` (실행 가능)

### 출력
- `outputs/test_detection_sample.png` (시각화)
- `outputs/initial_test_metrics.json` (성능 지표)

---

## 📝 변경 이력

### v1.0 (2025-01-19)
- ✅ HighlightDetector 모듈 구현
- ✅ HighlightEvaluator 구현
- ✅ ParameterOptimizer 프레임워크
- 🔍 초기 테스트 완료 (mIoU: 0.0)
- ⚠️ HSV 범위 불일치 문제 발견
- 📋 해결 방안 수립

---

## 🎓 학습 내용

### 1. HSV 색공간의 특성

**RGB vs HSV**:
- RGB: 빛의 삼원색, 직관적이지 않음
- HSV: Hue(색상), Saturation(채도), Value(명도)
- 색상 기반 객체 감지에 HSV가 훨씬 유리

**투명도의 영향**:
- Alpha blending은 RGB에서 작동
- HSV로 변환 시 Saturation이 감소
- 기본 HSV 범위 조정 필수

### 2. 모폴로지 연산의 중요성

**Opening (Erosion → Dilation)**:
- 작은 노이즈 제거
- 객체 경계 정제

**Closing (Dilation → Erosion)**:
- 작은 구멍 채우기
- 끊어진 영역 연결

**순서**: Closing → Opening (구멍 먼저 채우고 노이즈 제거)

### 3. 합성 데이터의 한계

**발견**:
- 합성 시 사용한 색상 ≠ 실제 렌더링 색상
- 투명도, 배경색, 압축 등이 영향
- Ground Truth는 정확하지만 **색상 가정이 틀렸음**

**교훈**:
- 합성 데이터 생성 후 실제 색상 측정 필수
- 알고리즘 파라미터를 데이터에 맞춰야 함

---

## 🔄 진행 상황

### 완료 ✅
- [x] HSV 기반 감지 알고리즘 구현
- [x] IoU 평가 시스템
- [x] 파라미터 최적화 프레임워크
- [x] 초기 테스트 및 문제 발견

### 진행 중 🔄
- [ ] HSV 범위 분석 및 조정
- [ ] 원본 이미지 최적화
- [ ] mIoU > 0.75 달성

### 예정 📋
- [ ] 증강 이미지 대응
- [ ] Tesseract OCR 통합
- [ ] OCR 성능 평가

---

## 🎯 목표 달성 현황

| 지표 | 목표 | 현재 | 달성률 |
|------|------|------|--------|
| mIoU | > 0.75 | 0.00 | 0% |
| Precision | > 0.80 | 0.00 | 0% |
| Recall | > 0.80 | 0.00 | 0% |
| F1-Score | > 0.80 | 0.00 | 0% |

**평가**: 초기 구현은 완료했으나 파라미터 최적화가 필수적

---

**문서 작성자**: AI Research Assistant
**최종 검토**: 2025-01-19
**다음 업데이트**: HSV 범위 최적화 완료 시
