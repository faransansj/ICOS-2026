# 컴퓨터 비전 기반 문서 하이라이트 텍스트 자동 추출 시스템 - 연구 워크플로우

## 프로젝트 개요

**연구 목표**: 형광펜으로 표시된 문서에서 하이라이트된 텍스트만을 자동으로 추출하는 OpenCV 기반 프로토타입 시스템 개발

**연구 기간**: 8주

**핵심 성능 목표**:
- 하이라이트 감지: mIoU > 0.75
- OCR 성능: CER < 5%
- End-to-End 정확도: > 80%

---

## Phase 1: 프로젝트 기반 설정 및 합성 데이터셋 생성 (1-2주)

### Week 1: 개발 환경 구축 및 데이터 생성 파이프라인 설계

#### Task 1.1: 프로젝트 환경 설정
**담당**: Backend/Infrastructure
**예상 시간**: 4시간
**의존성**: 없음

**구현 단계**:
1. **Python 가상환경 생성** (30분)
   - Python 3.8+ 가상환경 설정
   - requirements.txt 생성

2. **핵심 라이브러리 설치** (1시간)
   ```bash
   pip install opencv-python numpy pillow pytesseract albumentations pandas matplotlib scikit-learn
   ```
   - OpenCV 4.x
   - Tesseract OCR 엔진 설치 및 한글 언어팩
   - Albumentations (데이터 증강)

3. **프로젝트 디렉토리 구조 생성** (30분)
   ```
   text-highlight/
   ├── data/
   │   ├── synthetic/          # 합성 데이터
   │   ├── validation/         # 검증 데이터
   │   └── test/              # 테스트 데이터
   ├── src/
   │   ├── data_generator/    # 데이터 생성 모듈
   │   ├── highlight_detector/ # 하이라이트 감지 모듈
   │   ├── ocr_extractor/     # OCR 추출 모듈
   │   └── evaluation/        # 평가 모듈
   ├── notebooks/             # 실험 노트북
   ├── configs/               # 설정 파일
   └── outputs/               # 결과 저장
   ```

4. **Git 저장소 초기화** (30분)
   - .gitignore 설정
   - README.md 작성
   - 초기 커밋

**검증 기준**:
- [ ] 모든 라이브러리 정상 import
- [ ] Tesseract OCR 한글 텍스트 인식 테스트 성공
- [ ] 디렉토리 구조 생성 완료

---

#### Task 1.2: 텍스트 이미지 자동 생성 모듈 개발
**담당**: Backend Developer
**예상 시간**: 8시간
**의존성**: Task 1.1

**구현 단계**:
1. **텍스트 소스 데이터 준비** (1시간)
   - 다양한 분야의 한글 텍스트 샘플 수집 (교과서, 논문, 뉴스 등)
   - 문장 길이, 단어 밀도 다양화
   - JSON 형식으로 텍스트 데이터베이스 구축

2. **PIL 기반 텍스트 렌더링 엔진 구현** (3시간)
   ```python
   class TextImageGenerator:
       def __init__(self, fonts, text_sources):
           # 폰트 리스트, 텍스트 소스 초기화

       def generate_text_image(self, text, width=800, height=600):
           # 흰 배경에 검정 텍스트 렌더링
           # 다양한 폰트, 폰트 크기 적용
           # 줄 간격, 문단 간격 조절
   ```
   - 다양한 폰트 지원 (나눔고딕, 맑은 고딕 등)
   - 텍스트 레이아웃 알고리즘 (줄바꿈, 정렬)
   - 문단 단위 렌더링

3. **바운딩 박스 자동 생성** (2시간)
   - 단어/문장 단위 좌표 추출
   - Ground Truth annotation 자동 생성
   - JSON 형식 메타데이터 저장

4. **배치 생성 시스템** (2시간)
   - 100-200장 자동 생성 파이프라인
   - 다양한 레이아웃 변화 (2단, 3단, 일반)
   - 진행률 표시 및 로깅

**검증 기준**:
- [ ] 100장 이상 다양한 텍스트 이미지 생성
- [ ] Ground Truth 바운딩 박스 좌표 정확도 100%
- [ ] 메타데이터 JSON 파일 자동 생성

---

#### Task 1.3: 하이라이트 오버레이 시뮬레이션 모듈 개발
**담당**: Computer Vision Specialist
**예상 시간**: 10시간
**의존성**: Task 1.2

**구현 단계**:
1. **하이라이트 색상 정의** (1시간)
   ```python
   HIGHLIGHT_COLORS = {
       'yellow': (255, 255, 0),
       'green': (0, 255, 0),
       'pink': (255, 192, 203)
   }
   ```

2. **투명도 기반 오버레이 구현** (3시간)
   ```python
   class HighlightOverlay:
       def apply_highlight(self, image, bbox, color, alpha=0.3):
           # OpenCV addWeighted를 이용한 투명 하이라이트
           # 불규칙한 경계선 시뮬레이션 (Gaussian noise)
   ```
   - Alpha blending (투명도 0.2-0.4)
   - 경계선 불규칙성 추가 (실제 형광펜 효과)

3. **랜덤 하이라이트 선택 알고리즘** (2시간)
   - 전체 텍스트의 20-40% 랜덤 선택
   - 문장 단위/단어 단위 하이라이트
   - 연속된 문장 하이라이트 확률 조절

4. **Ground Truth 업데이트** (2시간)
   - 하이라이트된 영역의 bbox, text, color 정보 저장
   - Validation을 위한 완벽한 annotation 생성

5. **시각적 검증 도구** (2시간)
   - 하이라이트 이미지와 Ground Truth 시각화
   - 랜덤 샘플 확인 스크립트

**검증 기준**:
- [ ] 3가지 색상 하이라이트 정상 작동
- [ ] 투명도 및 불규칙성 실제 형광펜과 유사
- [ ] Ground Truth annotation 100% 정확

---

### Week 2: 데이터 증강 및 데이터셋 완성

#### Task 2.1: Albumentations 기반 데이터 증강 파이프라인
**담당**: ML Engineer
**예상 시간**: 6시간
**의존성**: Task 1.3

**구현 단계**:
1. **증강 변환 정의** (2시간)
   ```python
   transform = A.Compose([
       A.RandomBrightnessContrast(p=0.5),
       A.GaussNoise(p=0.3),
       A.Rotate(limit=5, p=0.5),
       A.Perspective(scale=0.05, p=0.3),
       A.MotionBlur(p=0.2)
   ], bbox_params=A.BboxParams(format='pascal_voc'))
   ```
   - 조명 변화 (밝기, 대비)
   - 노이즈 추가 (Gaussian, Salt-pepper)
   - 회전 및 원근 변환
   - 모션 블러 (스캔/촬영 흔들림)

2. **바운딩 박스 보존 증강** (2시간)
   - Albumentations bbox 변환 적용
   - 증강 후 좌표 자동 업데이트

3. **배치 증강 실행** (2시간)
   - 원본 200장 → 증강 400-600장
   - 진행률 모니터링
   - 증강 전후 비교 샘플 저장

**검증 기준**:
- [ ] 증강 후 바운딩 박스 좌표 일치율 > 95%
- [ ] 다양한 조명/회전 조건 샘플 생성
- [ ] 총 400장 이상 데이터셋 확보

---

#### Task 2.2: 데이터셋 분할 및 메타데이터 정리
**담당**: Data Engineer
**예상 시간**: 4시간
**의존성**: Task 2.1

**구현 단계**:
1. **데이터 분할** (1시간)
   - Validation: 30% (~150장)
   - Test: 70% (~350장)
   - Stratified split (색상별, 텍스트 밀도별 균등 분배)

2. **메타데이터 통합** (2시간)
   - 단일 JSON 파일로 annotation 통합
   - COCO 형식 호환 변환
   ```json
   {
       "images": [...],
       "annotations": [
           {
               "id": 1,
               "image_id": 1,
               "bbox": [x, y, w, h],
               "text": "하이라이트된 텍스트",
               "color": "yellow",
               "category": "highlight"
           }
       ]
   }
   ```

3. **데이터셋 통계 분석** (1시간)
   - 색상별 분포
   - 텍스트 길이 분포
   - 하이라이트 영역 크기 분포
   - 시각화 및 리포트 생성

**검증 기준**:
- [ ] Validation/Test 분할 완료
- [ ] 메타데이터 JSON 스키마 검증
- [ ] 데이터셋 통계 리포트 생성

---

## Phase 2: 하이라이트 감지 및 OCR 모듈 개발 (3-4주)

### Week 3: 하이라이트 영역 감지 모듈 개발

#### Task 3.1: HSV 색공간 기반 색상 마스크 생성
**담당**: Computer Vision Engineer
**예상 시간**: 8시간
**의존성**: Task 2.2

**구현 단계**:
1. **HSV 색상 범위 정의** (2시간)
   ```python
   HSV_RANGES = {
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
   - Validation 데이터로 색상 범위 튜닝
   - 조명 변화에 강인한 범위 설정

2. **마스크 생성 및 후처리** (4시간)
   ```python
   class HighlightDetector:
       def detect_highlights(self, image):
           hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
           mask = cv2.inRange(hsv, lower, upper)

           # 모폴로지 연산
           kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
           mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
           mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

           return mask
   ```
   - 모폴로지 연산 (opening, closing)
   - 노이즈 제거

3. **컨투어 추출 및 바운딩 박스 생성** (2시간)
   - `cv2.findContours`를 이용한 영역 검출
   - 최소 면적 필터링 (작은 노이즈 제거)
   - 바운딩 박스 좌표 추출

**검증 기준**:
- [ ] 각 색상별 마스크 생성 성공
- [ ] 노이즈 필터링 후 깨끗한 영역 검출
- [ ] Validation 데이터에서 mIoU > 0.70 (초기 목표)

---

#### Task 3.2: 하이라이트 감지 성능 평가 시스템
**담당**: ML Engineer
**예상 시간**: 6시간
**의존성**: Task 3.1

**구현 단계**:
1. **IoU 계산 함수 구현** (2시간)
   ```python
   def calculate_iou(bbox1, bbox2):
       # Intersection over Union 계산
       x1, y1, w1, h1 = bbox1
       x2, y2, w2, h2 = bbox2

       intersection = calculate_intersection_area(bbox1, bbox2)
       union = w1*h1 + w2*h2 - intersection

       return intersection / union
   ```

2. **mIoU, Precision, Recall, F1 계산** (2시간)
   - Ground Truth와 예측 결과 매칭
   - 색상별 성능 분리 평가

3. **평가 리포트 자동 생성** (2시간)
   - Validation 데이터 전체 평가
   - 색상별, 조명별 성능 분석
   - 실패 케이스 시각화

**검증 기준**:
- [ ] mIoU 계산 정확도 검증
- [ ] Validation 데이터 평가 리포트 생성
- [ ] 실패 케이스 분석 완료

---

#### Task 3.3: 하이라이트 감지 하이퍼파라미터 최적화
**담당**: ML Engineer
**예상 시간**: 8시간
**의존성**: Task 3.2

**구현 단계**:
1. **그리드 서치 파라미터 정의** (2시간)
   - HSV 색상 범위 (±5 단위 조정)
   - 모폴로지 커널 크기 (3x3, 5x5, 7x7)
   - 최소 컨투어 면적 임계값

2. **자동 튜닝 스크립트** (4시간)
   ```python
   best_params = grid_search(
       param_grid={
           'hsv_yellow_lower_h': [18, 20, 22],
           'kernel_size': [3, 5, 7],
           'min_area': [50, 100, 150]
       },
       metric='mIoU'
   )
   ```

3. **최적 파라미터 검증** (2시간)
   - Validation 데이터에서 mIoU > 0.75 달성
   - Test 데이터에서 일반화 성능 확인

**검증 기준**:
- [ ] Validation mIoU > 0.75
- [ ] 색상별 F1-Score > 0.80
- [ ] 최적 파라미터 config 파일 저장

---

### Week 4: OCR 텍스트 추출 모듈 개발

#### Task 4.1: Tesseract OCR 텍스트 추출 엔진
**담당**: Backend Developer
**예상 시간**: 6시간
**의존성**: Task 3.1

**구현 단계**:
1. **Tesseract 설정 최적화** (2시간)
   ```python
   custom_config = r'--oem 3 --psm 6 -l kor+eng'
   text = pytesseract.image_to_string(image, config=custom_config)
   ```
   - 한글+영어 동시 인식
   - PSM 모드 선택 (6: 단일 블록 텍스트)

2. **바운딩 박스 기반 텍스트 추출** (2시간)
   ```python
   def extract_text_with_boxes(image):
       data = pytesseract.image_to_data(image, output_type=Output.DICT)

       results = []
       for i, text in enumerate(data['text']):
           if text.strip():
               x, y, w, h = data['left'][i], data['top'][i],
                            data['width'][i], data['height'][i]
               results.append({
                   'text': text,
                   'bbox': [x, y, w, h],
                   'confidence': data['conf'][i]
               })
       return results
   ```

3. **전처리 파이프라인** (2시간)
   - 이진화 (Otsu's method)
   - 노이즈 제거
   - 대비 향상

**검증 기준**:
- [ ] 한글 텍스트 정상 인식
- [ ] 바운딩 박스 좌표 추출 성공
- [ ] Confidence score 계산 정상 작동

---

#### Task 4.2: OCR 성능 평가 시스템
**담당**: ML Engineer
**예상 시간**: 6시간
**의존성**: Task 4.1

**구현 단계**:
1. **CER/WER 계산 함수** (3시간)
   ```python
   def calculate_cer(ground_truth, predicted):
       # Levenshtein distance 기반 Character Error Rate
       distance = edit_distance(ground_truth, predicted)
       return distance / len(ground_truth)

   def calculate_wer(ground_truth, predicted):
       # Word Error Rate 계산
   ```

2. **OCR 평가 리포트** (2시간)
   - Validation 데이터 전체 평가
   - 문자 단위/단어 단위 정확도
   - 실패 케이스 분석

3. **Confidence threshold 최적화** (1시간)
   - 낮은 confidence 결과 필터링
   - Precision-Recall 트레이드오프 분석

**검증 기준**:
- [ ] CER < 5% 달성
- [ ] WER < 10% 달성
- [ ] Confidence threshold 최적값 도출

---

## Phase 3: 통합 시스템 구축 및 최적화 (5-6주)

### Week 5: End-to-End 파이프라인 통합

#### Task 5.1: IoU 기반 하이라이트-텍스트 매칭 모듈
**담당**: Backend Developer
**예상 시간**: 8시간
**의존성**: Task 3.3, Task 4.2

**구현 단계**:
1. **매칭 알고리즘 구현** (4시간)
   ```python
   def match_highlights_to_text(highlight_boxes, ocr_boxes, iou_threshold=0.5):
       matched_results = []

       for h_box in highlight_boxes:
           best_match = None
           best_iou = 0

           for ocr_box in ocr_boxes:
               iou = calculate_iou(h_box['bbox'], ocr_box['bbox'])
               if iou > best_iou and iou > iou_threshold:
                   best_iou = iou
                   best_match = ocr_box

           if best_match:
               matched_results.append({
                   'text': best_match['text'],
                   'bbox': h_box['bbox'],
                   'color': h_box['color'],
                   'iou': best_iou
               })

       return matched_results
   ```

2. **IoU 임계값 최적화** (2시간)
   - Validation 데이터로 0.3~0.7 범위 탐색
   - Precision-Recall 곡선 분석

3. **중복 매칭 처리** (2시간)
   - 동일 OCR 박스에 여러 하이라이트 매칭 시 처리
   - 최대 IoU 우선 선택

**검증 기준**:
- [ ] IoU 기반 매칭 정확도 > 90%
- [ ] 최적 IoU 임계값 도출
- [ ] 중복 매칭 케이스 정상 처리

---

#### Task 5.2: End-to-End 통합 파이프라인 구현
**담당**: System Architect
**예상 시간**: 10시간
**의존성**: Task 5.1

**구현 단계**:
1. **메인 파이프라인 클래스** (4시간)
   ```python
   class HighlightTextExtractor:
       def __init__(self):
           self.highlight_detector = HighlightDetector()
           self.ocr_extractor = OCRExtractor()

       def process_image(self, image_path):
           # 1. 이미지 로드 및 전처리
           image = cv2.imread(image_path)
           preprocessed = self.preprocess(image)

           # 2. 하이라이트 영역 감지
           highlight_boxes = self.highlight_detector.detect(preprocessed)

           # 3. OCR 텍스트 추출
           ocr_boxes = self.ocr_extractor.extract(preprocessed)

           # 4. 하이라이트-텍스트 매칭
           results = self.match(highlight_boxes, ocr_boxes)

           return results
   ```

2. **에러 핸들링 및 로깅** (2시간)
   - 각 단계별 예외 처리
   - 진행 상황 로깅
   - 중간 결과 저장 옵션

3. **배치 처리 기능** (2시간)
   - 다중 이미지 병렬 처리
   - 진행률 표시

4. **결과 저장 포맷** (2시간)
   - JSON 형식 출력
   - CSV 형식 출력
   - 시각화 이미지 생성 (하이라이트 텍스트 표시)

**검증 기준**:
- [ ] 단일 이미지 처리 성공
- [ ] 배치 처리 (100장) 정상 작동
- [ ] 다양한 출력 포맷 지원

---

### Week 6: 시스템 최적화 및 성능 튜닝

#### Task 6.1: 파라미터 통합 최적화
**담당**: ML Engineer
**예상 시간**: 12시간
**의존성**: Task 5.2

**구현 단계**:
1. **전체 파라미터 그리드 서치** (6시간)
   - HSV 색상 범위
   - 모폴로지 커널
   - IoU 임계값
   - OCR confidence threshold
   - Validation 데이터로 최적화

2. **색상별 개별 최적화** (4시간)
   - 노랑/초록/분홍 각각 최적 파라미터
   - 색상별 성능 균형 조정

3. **조명 조건별 파라미터 분석** (2시간)
   - 밝은/어두운 조명에서의 성능
   - Adaptive parameter 전략 고려

**검증 기준**:
- [ ] Validation 데이터에서 End-to-End Accuracy > 80%
- [ ] 색상별 F1-Score 균형 (편차 < 0.1)

---

#### Task 6.2: 성능 병목 지점 분석 및 최적화
**담당**: Performance Engineer
**예상 시간**: 6시간
**의존성**: Task 6.1

**구현 단계**:
1. **프로파일링** (2시간)
   - 각 모듈별 처리 시간 측정
   - 메모리 사용량 분석

2. **최적화 적용** (3시간)
   - NumPy vectorization
   - OpenCV 함수 최적화
   - 불필요한 복사 제거

3. **병렬 처리 구현** (1시간)
   - 다중 이미지 처리 시 멀티프로세싱
   - ThreadPoolExecutor 활용

**검증 기준**:
- [ ] 단일 이미지 처리 시간 < 2초
- [ ] 배치 처리 속도 > 50 images/min

---

## Phase 4: 평가 및 분석 (7주)

### Week 7: 최종 성능 평가 및 실패 사례 분석

#### Task 7.1: Test 데이터셋 전체 평가
**담당**: QA Engineer
**예상 시간**: 8시간
**의존성**: Task 6.2

**구현 단계**:
1. **Test 데이터 평가 실행** (2시간)
   - 350장 Test 이미지 처리
   - 결과 자동 저장

2. **평가 지표 계산** (3시간)
   - **하이라이트 감지**: mIoU, Precision, Recall, F1
   - **OCR 성능**: CER, WER
   - **통합 성능**: End-to-End Accuracy, F1-Score

3. **상세 분석 리포트 생성** (3시간)
   - 색상별 성능 비교
   - 조명별 성능 비교
   - 텍스트 밀도별 성능 비교
   - Confusion matrix 및 시각화

**검증 기준**:
- [ ] Test 데이터 평가 완료
- [ ] 모든 지표 목표치 달성 확인
- [ ] 상세 분석 리포트 작성

---

#### Task 7.2: 실패 케이스 분석 및 개선 방안 도출
**담당**: Analyzer
**예상 시간**: 8시간
**의존성**: Task 7.1

**구현 단계**:
1. **실패 케이스 수집** (2시간)
   - IoU < 0.5인 케이스
   - CER > 10%인 케이스
   - False Positive/Negative 케이스

2. **실패 원인 분류** (4시간)
   - 하이라이트 감지 실패 (색상 범위, 조명)
   - OCR 인식 실패 (폰트, 해상도)
   - 매칭 실패 (IoU 임계값)

3. **개선 방안 제안** (2시간)
   - 추가 전처리 기법
   - 딥러닝 기반 하이라이트 감지 고려
   - OCR 모델 fine-tuning 방안

**검증 기준**:
- [ ] 실패 케이스 카테고리별 분류 완료
- [ ] 개선 방안 문서화
- [ ] 향후 연구 방향 제시

---

#### Task 7.3: 성능 벤치마크 및 비교 분석
**담당**: Researcher
**예상 시간**: 6시간
**의존성**: Task 7.2

**구현 단계**:
1. **기존 방법과의 비교** (3시간)
   - 단순 색상 필터링 vs. 본 시스템
   - 전체 OCR vs. 선별적 OCR
   - 성능/속도 트레이드오프 분석

2. **학술적 기여 정리** (2시간)
   - 하이라이트 감지 + OCR 통합의 novelty
   - 합성 데이터 생성 방법론의 재사용성

3. **실용적 가치 평가** (1시간)
   - 실제 사용 시나리오 적용 가능성
   - 확장 가능성 (손글씨, 다양한 문서 형식)

**검증 기준**:
- [ ] 기존 방법 대비 성능 우위 입증
- [ ] 학술적/실용적 기여 명확화

---

## Phase 5: 결과 정리 및 문서화 (8주)

### Week 8: 최종 보고서 작성 및 코드 정리

#### Task 8.1: 연구 보고서 작성
**담당**: Technical Writer
**예상 시간**: 12시간
**의존성**: Task 7.3

**구현 단계**:
1. **보고서 구조** (2시간)
   ```
   1. 서론 (연구 배경, 필요성, 목표)
   2. 관련 연구 (OCR, 하이라이트 감지)
   3. 방법론
      3.1 시스템 파이프라인
      3.2 합성 데이터셋 생성
      3.3 하이라이트 감지 알고리즘
      3.4 OCR 텍스트 추출
      3.5 매칭 알고리즘
   4. 실험 결과
      4.1 데이터셋 구성
      4.2 평가 지표 및 결과
      4.3 비교 분석
   5. 분석 및 논의
      5.1 성공 사례
      5.2 실패 사례 분석
      5.3 한계점
   6. 결론 및 향후 연구
   ```

2. **내용 작성** (8시간)
   - 각 섹션별 상세 내용 작성
   - 표, 그림, 수식 삽입
   - 참고문헌 정리

3. **검토 및 수정** (2시간)
   - 논리적 흐름 확인
   - 문법 및 표현 개선

**검증 기준**:
- [ ] 보고서 전체 구조 완성
- [ ] 모든 실험 결과 포함
- [ ] 참고문헌 정리 완료

---

#### Task 8.2: 코드 정리 및 문서화
**담당**: Software Engineer
**예상 시간**: 8시간
**의존성**: Task 7.3

**구현 단계**:
1. **코드 리팩토링** (4시간)
   - PEP 8 스타일 준수
   - Docstring 추가
   - 불필요한 코드 제거

2. **README.md 작성** (2시간)
   - 프로젝트 소개
   - 설치 방법
   - 사용 예제
   - 디렉토리 구조 설명

3. **API 문서 생성** (2시간)
   - Sphinx를 이용한 자동 문서화
   - 주요 클래스 및 함수 설명

**검증 기준**:
- [ ] 코드 스타일 일관성
- [ ] README.md 완성도
- [ ] API 문서 생성 완료

---

#### Task 8.3: 최종 결과물 패키징 및 데모
**담당**: Project Manager
**예상 시간**: 6시간
**의존성**: Task 8.1, Task 8.2

**구현 단계**:
1. **결과물 정리** (2시간)
   - 코드 저장소 (GitHub)
   - 데이터셋 샘플
   - 평가 결과 리포트
   - 연구 보고서 PDF

2. **데모 스크립트 작성** (2시간)
   - 샘플 이미지 입력 → 결과 출력
   - Jupyter Notebook 데모

3. **최종 검토** (2시간)
   - 모든 목표 달성 확인
   - 체크리스트 검증

**검증 기준**:
- [ ] GitHub 저장소 공개 준비
- [ ] 데모 스크립트 정상 작동
- [ ] 모든 산출물 완성

---

## 평가 기준 및 목표 달성 체크리스트

### 하이라이트 감지 성능
- [ ] mIoU > 0.75 (Validation)
- [ ] mIoU > 0.75 (Test)
- [ ] Precision > 0.80
- [ ] Recall > 0.80
- [ ] F1-Score > 0.80

### OCR 성능
- [ ] CER < 5% (Validation)
- [ ] CER < 5% (Test)
- [ ] WER < 10%

### End-to-End 통합 성능
- [ ] End-to-End Accuracy > 80% (Validation)
- [ ] End-to-End Accuracy > 80% (Test)
- [ ] F1-Score > 0.75

### 시스템 성능
- [ ] 단일 이미지 처리 시간 < 2초
- [ ] 배치 처리 > 50 images/min

### 데이터셋
- [ ] 합성 이미지 200장 생성
- [ ] 증강 데이터 400장 생성
- [ ] Ground Truth annotation 100% 정확

### 문서화
- [ ] 연구 보고서 작성 완료
- [ ] README.md 작성
- [ ] API 문서 생성
- [ ] 코드 주석 및 Docstring 완성

---

## 리스크 관리

### 기술적 리스크
| 리스크 | 영향도 | 발생확률 | 완화 방안 |
|--------|--------|----------|-----------|
| HSV 색상 범위가 조명에 민감 | 높음 | 중간 | Lab 색공간 추가 탐색, Adaptive threshold |
| OCR 한글 인식률 저조 | 높음 | 낮음 | 전처리 강화, EasyOCR 대안 고려 |
| IoU 매칭 정확도 부족 | 중간 | 낮음 | IoU threshold 최적화, 다중 매칭 전략 |
| 합성 데이터와 실제 데이터 괴리 | 중간 | 중간 | 증강 강화, 실제 스캔 샘플 추가 |

### 일정 리스크
| 리스크 | 완화 방안 |
|--------|-----------|
| 데이터 생성 지연 | Week 1에 우선 집중, 병렬 작업 최소화 |
| 파라미터 튜닝 장기화 | 자동화 스크립트 사용, 조기 중단 조건 설정 |
| 예상치 못한 버그 | 단위 테스트 작성, 코드 리뷰 |

---

## 참고 자료

### 핵심 논문
- Tan, R.T., et al. (2004). "Separating Reflection Components Based on Chromaticity and Noise Analysis." IEEE TPAMI.
- Rafael C. (2021). "Evaluate OCR Output Quality with Character Error Rate and Word Error Rate." Towards Data Science.
- Rezatofighi, H., et al. (2019). "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression." CVPR.

### 기술 문서
- OpenCV Documentation: https://docs.opencv.org/
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- Albumentations: https://albumentations.ai/docs/

---

## 버전 정보

**문서 버전**: v1.0
**작성일**: 2025년 1월
**연구 기간**: 8주
**상태**: 워크플로우 확정
