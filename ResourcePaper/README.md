# 논문 작성 리소스 가이드 (Research Paper Resources Guide)

## 📚 개요

이 디렉토리는 **"컴퓨터 비전 기반 문서 하이라이트 텍스트 자동 추출 시스템"** 논문 작성을 위한 모든 리소스를 포함합니다.

**프로젝트 정보**:
- **연구 주제**: 하이라이트 텍스트 자동 추출 시스템
- **주요 성과**: OCR 정확도 95.30%, CER 4.70%, mIoU 0.8222
- **연구 기간**: 2025-01-15 ~ 2025-10-19
- **시스템 버전**: 1.0.0

---

## 📂 디렉토리 구조

```
ResourcePaper/
├── README.md                           # 이 파일
├── framework/                          # 논문 구조 프레임워크
│   └── 01_paper_structure.md          # 전체 논문 구조 및 섹션 가이드
├── data/                               # 성능 데이터 및 메트릭
│   └── performance_metrics.md         # 모든 성능 지표 및 통계
├── architecture/                       # 시스템 아키텍처
│   └── system_pipeline.md             # 3단계 파이프라인 상세 설명
├── methodology/                        # 방법론 및 알고리즘
│   └── algorithms_code.md             # 핵심 알고리즘 및 수식
├── results/                            # 실험 결과
│   └── experimental_results.md        # 실험 설정, 결과, 분석
└── figures/                            # 그림 및 표
    └── figure_descriptions.md         # 모든 그림/표 설명

총 파일: 7개
총 크기: ~150 KB (텍스트)
```

---

## 📖 각 파일 사용법

### 1. framework/01_paper_structure.md

**용도**: 논문 전체 구조 설계

**포함 내용**:
- 10개 섹션별 구성 (초록 ~ 부록)
- 각 섹션 분량 및 하위 구조
- 그림/표 목록 (15-20개)
- 작성 스타일 가이드
- 체크리스트

**활용 방법**:
1. 논문 작성 시작 전 전체 구조 파악
2. 각 섹션별 필수 포함 내용 확인
3. 분량 배분 참고 (총 20-25페이지)
4. 작성 완료 후 체크리스트로 검증

---

### 2. data/performance_metrics.md

**용도**: 모든 성능 지표 및 실험 데이터

**포함 내용**:
- 최종 성능 요약 (mIoU, CER, 정확도)
- 단계별 OCR 성능 개선 (9단계)
- 색상별 성능 분석
- 오류 분석 (14개 오류 상세)
- 처리 성능 (시간, 메모리)
- 데이터셋 통계
- 비교 분석 (기존 연구)
- 통계적 유의성 검증

**활용 방법**:
1. Section 6 (실험 결과) 작성 시 참조
2. 표 작성을 위한 원본 데이터
3. 그래프 생성을 위한 수치 데이터
4. Discussion에서 인용할 구체적 수치

**주요 테이블**:
- Table 5: OCR 성능 개선 과정 (9단계)
- Table 6: 최종 성능 메트릭
- Table 7: 색상별 OCR 성능

---

### 3. architecture/system_pipeline.md

**용도**: 시스템 설계 및 파이프라인 설명

**포함 내용**:
- 3단계 파이프라인 전체 구조
- Stage 1: 하이라이트 검출 상세
  - HSV 변환, 마스킹, 형태학적 연산
- Stage 2: OCR 텍스트 추출
  - Tesseract 설정, Multi-PSM 전략
- Stage 3: 후처리
  - 한국어 공백 제거, 조사 복원, 문자 치환
- 출력 포맷 (JSON, CSV, TXT, 시각화)
- 핵심 클래스 구조
- 파이프라인 성능 특성

**활용 방법**:
1. Section 3 (시스템 아키텍처) 작성
2. Section 4 (방법론) 중 파이프라인 부분
3. Figure 1 (전체 시스템 아키텍처) 참조
4. 코드 스니펫 복사 (필요 시)

**주요 다이어그램**:
- 전체 파이프라인 ASCII 아트
- 각 단계별 상세 프로세스

---

### 4. methodology/algorithms_code.md

**용도**: 핵심 알고리즘 및 수학적 모델

**포함 내용**:
- Algorithm 1: HSV 색공간 기반 하이라이트 검출
- Algorithm 2: 다중 PSM 모드 OCR
- Algorithm 3: 한국어 공백 제거 (재귀)
- Algorithm 4: 중복 텍스트 제거
- 성능 평가 메트릭 (CER, mIoU)
- 실제 Python 코드 구현
- 수학적 수식 및 모델
- 복잡도 분석

**활용 방법**:
1. Section 4 (방법론) 작성 시 참조
2. 알고리즘 수도코드 복사
3. 수학적 수식 LaTeX 변환
4. 코드 재현성 보장

**주요 수식**:
- HSV 변환: `H = arctan2(...)`
- CER: `(S + D + I) / N`
- IoU: `Area(∩) / Area(∪)`
- mIoU: `(1/N) × Σ IoU_i`

---

### 5. results/experimental_results.md

**용도**: 실험 설정, 결과, 분석

**포함 내용**:
- 실험 환경 (H/W, S/W)
- 데이터셋 구성
- Phase 2-1: 하이라이트 검출 결과
- Phase 2-2: OCR 성능 개선 과정
- 오류 분석 (14개 오류 상세)
- 처리 성능 분석
- 통계적 검증 (t-검정)
- 비교 분석
- 목표 달성도
- 주요 발견 사항

**활용 방법**:
1. Section 6 (실험 결과) 전체 작성
2. Section 7 (Discussion) 분석 부분
3. 성공/실패 사례 인용
4. 통계적 유의성 증명

**주요 분석**:
- 단계별 성능 변화 (Table 5)
- 색상별 성능 차이
- t-검정 결과 (p < 0.001)
- 5가지 실패 사례 상세

---

### 6. figures/figure_descriptions.md

**용도**: 모든 그림 및 표 설명

**포함 내용**:
- 10개 그림 상세 설명
  - Figure 1: 전체 시스템 아키텍처
  - Figure 2: HSV 색공간 시각화
  - Figure 3: 하이라이트 검출 프로세스
  - Figure 4: OCR 성능 개선 과정
  - Figure 5: 색상별 성능 비교
  - Figure 6: 후처리 전후 비교
  - Figure 7: 처리 시간 분석
  - Figure 8: 배치 처리 확장성
  - Figure 9: 오류 분석 차트
  - Figure 10: 실제 검출 결과 예시
- 10개 표 상세 설명
- 그림/표 생성 가이드
- 스타일 가이드

**활용 방법**:
1. 각 그림/표 생성 시 참조
2. 캡션 작성 템플릿
3. 크기 및 배치 결정
4. 생성 도구 선택

---

## 🚀 논문 작성 워크플로우

### Phase 1: 구조 설계 (1일)

1. `framework/01_paper_structure.md` 읽기
2. 논문 전체 구조 확정
3. 각 섹션별 분량 배분
4. 그림/표 위치 결정

### Phase 2: 데이터 준비 (1일)

1. `data/performance_metrics.md`에서 필요한 수치 추출
2. `results/experimental_results.md`에서 결과 정리
3. 표 초안 작성 (10개)
4. 그래프 데이터 준비

### Phase 3: 본문 작성 (5-7일)

**Section 1-2: 서론** (0.5일)
- 연구 배경 및 목적
- 관련 연구 조사

**Section 3-4: 시스템 및 방법론** (2일)
- `architecture/system_pipeline.md` 참조
- `methodology/algorithms_code.md` 참조
- 알고리즘 수도코드 작성
- 수학적 수식 LaTeX 변환

**Section 5-6: 실험 및 결과** (2일)
- `results/experimental_results.md` 참조
- `data/performance_metrics.md`에서 표 작성
- 그림 생성 (Figure 4, 5, 7, 8, 9)

**Section 7-8: 토의 및 결론** (1일)
- 성공 요인 분석
- 한계점 및 향후 연구
- 연구 기여 요약

**Section 9-10: 참고문헌 및 부록** (0.5일)
- 참고문헌 정리 (10-15개)
- 부록 작성 (HSV 범위, 설정 등)

### Phase 4: 그림 생성 (2-3일)

1. Figure 1: 시스템 아키텍처 (draw.io)
2. Figure 2: HSV 색공간 (Plotly 3D)
3. Figure 3: 검출 프로세스 (OpenCV + PIL)
4. Figure 4: 성능 개선 (Matplotlib)
5. Figure 5: 색상별 비교 (Seaborn)
6. Figure 6: 전후 비교 (표 형식)
7. Figure 7: 처리 시간 (파이 차트)
8. Figure 8: 확장성 (라인 차트)
9. Figure 9: 오류 분석 (복합 차트)
10. Figure 10: 실제 결과 (이미지 주석)

### Phase 5: 검토 및 수정 (2-3일)

1. `framework/01_paper_structure.md` 체크리스트 확인
2. 논리 흐름 점검
3. 수치 정확성 검증
4. 맞춤법 및 문법 검토
5. 참고문헌 형식 통일
6. 최종 포맷팅

**총 예상 기간**: 10-15일

---

## 📊 핵심 수치 퀵 레퍼런스

### 최종 성능
- **OCR 정확도**: 95.30%
- **CER**: 4.70%
- **mIoU**: 0.8222
- **총 오류**: 14/298자
- **오류 감소율**: 94.4% (241개 → 14개)

### 데이터셋
- **총 이미지**: 600개
- **분할**: 훈련 360 / 검증 90 / 테스트 150
- **하이라이트**: 2,400개
- **검증 문자 수**: 298자

### 처리 성능
- **단일 이미지**: 0.82초
- **배치 (50개)**: 41.2초 (순차), 12.1초 (병렬)
- **메모리**: 평균 150MB

### 목표 달성
- mIoU: 109.6% (목표 대비)
- 정확도: 100.3% (목표 대비)
- CER: 94.0% (목표 달성)

---

## 🔗 관련 파일 링크

### 프로젝트 파일
- 메인 논문: `../outputs/research_paper.md`
- 최종 성능 보고서: `../outputs/final_performance_report.md`
- OCR 개선 요약: `../outputs/ocr_improvement_summary.md`

### 소스 코드
- 파이프라인: `../src/pipeline.py`
- 하이라이트 검출: `../src/highlight_detector.py`
- OCR 엔진: `../src/ocr/ocr_engine.py`
- CLI: `../extract_highlights.py`

### 설정 파일
- HSV 범위: `../configs/optimized_hsv_ranges.json`

---

## 📝 작성 팁

### 1. 수식 작성
LaTeX 수식을 사용하여 수학적 모델 표현:
```latex
\text{CER} = \frac{S + D + I}{N}
```

### 2. 표 작성
Markdown 표를 Word/LaTeX로 변환:
- 온라인 변환기 사용
- 또는 직접 복사-붙여넣기

### 3. 그림 캡션
```
Figure X: [영문 제목]
[상세 설명 2-3문장]
```

### 4. 인용
논문 내 수치 인용 시:
```
"본 시스템은 95.30%의 OCR 정확도를 달성하였다 (Table 6)."
"하이라이트 검출 mIoU는 0.8222로 목표 대비 9.6% 향상되었다 (Figure 5)."
```

### 5. 코드 블록
알고리즘 수도코드는 들여쓰기 유지:
```
Algorithm 1: Highlight Detection
Input: RGB image I
Output: Bounding boxes B
1. Convert I to HSV
2. For each color c:
    a. Create mask
    b. Apply morphology
    ...
```

---

## ✅ 최종 체크리스트

### 내용 완성도
- [ ] 모든 섹션 작성 완료 (1-10)
- [ ] 그림 10개 생성 및 삽입
- [ ] 표 10개 작성 및 포맷팅
- [ ] 참고문헌 10-15개
- [ ] 부록 A-F 작성

### 형식 준수
- [ ] 페이지 번호 삽입
- [ ] 섹션 번호 일관성 (1, 1.1, 1.1.1)
- [ ] 그림/표 캡션 및 번호
- [ ] 참고문헌 형식 통일 (IEEE/APA)
- [ ] 키워드 5-7개

### 품질 검토
- [ ] 맞춤법 및 문법 (한글/영문)
- [ ] 용어 일관성 (OCR = 광학 문자 인식)
- [ ] 수치 정확성 검증
- [ ] 논리 흐름 점검
- [ ] 중복 내용 제거

### 재현성
- [ ] 실험 환경 명시
- [ ] 파라미터 전부 기록
- [ ] 데이터셋 분할 명시
- [ ] 코드 저장소 링크 (GitHub)

---

## 📞 문의 및 지원

**프로젝트 디렉토리**: `/Users/midori/Research/Text-Highlight/`
**논문 리소스 디렉토리**: `/Users/midori/Research/Text-Highlight/ResourcePaper/`

**생성 일자**: 2025-10-19
**최종 업데이트**: 2025-10-19
**버전**: 1.0

---

**Good luck with your paper! 📚✨**
