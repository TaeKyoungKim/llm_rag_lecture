# FAISS HNSW 시스템 사용 가이드

## 개요
LangChain을 활용한 FAISS HNSW 벡터 인덱싱 시스템으로, 기술적 분석 PDF 문서를 벡터화하여 효율적인 의미론적 검색을 제공합니다.

## 파일 구조
```
DocumentsLoader/
├── data/
│   └── 기술적차트분석이론및방법.pdf  # 분석 대상 PDF
├── faiss_index_complete/              # 완전한 FAISS 인덱스
├── faiss_index/                       # 간단한 테스트 인덱스
├── search_results/                    # 검색 결과 JSON 파일들
├── faiss_hnsw_indexing_system.py     # 기본 FAISS 시스템
├── simple_faiss_test.py              # 간단한 테스트 시스템
├── complete_faiss_system.py          # 완전한 FAISS 시스템
└── README_FAISS_System.md            # 이 파일
```

## 설치 및 실행

### 1. 필요한 패키지 설치
```bash
uv add langchain_community faiss-cpu sentence-transformers pypdf pandas
```

### 2. 간단한 테스트 시스템 실행
```bash
uv run DocumentsLoader/simple_faiss_test.py
```

### 3. 완전한 FAISS 시스템 실행
```bash
uv run DocumentsLoader/complete_faiss_system.py
```

## 주요 기능

### 1. PDF 문서 처리
- **문서 로드**: PyPDFLoader를 통한 PDF 텍스트 추출
- **청크 분할**: RecursiveCharacterTextSplitter로 최적화된 청크 생성
- **메타데이터 관리**: 페이지 정보, 청크 ID, 기술적 내용 포함 여부 등

### 2. 벡터 인덱싱
- **임베딩 모델**: HuggingFace sentence-transformers (한국어 최적화)
- **인덱스 타입**: FAISS HNSW (Hierarchical Navigable Small World)
- **저장 및 로드**: 인덱스 재사용 가능

### 3. 검색 기능

#### 의미론적 검색
```python
# 의미론적 검색 예시
results = system.semantic_search("RSI 상대강도지수 분석 방법")
```

#### 기술적 분석 특화 검색
```python
# 기술적 분석 검색 예시
results = system.technical_search("RSI")  # RSI 관련 내용 검색
```

#### 지원하는 기술적 분석 지표
1. **RSI** - 상대강도지수 (과매수/과매도 판단)
2. **MACD** - 이동평균수렴확산 (추세 추종)
3. **볼린저밴드** - 변동성 측정
4. **이동평균선** - 추세 파악
5. **스토캐스틱** - 오실레이터
6. **일목균형표** - 일본식 분석
7. **피보나치** - 되돌림 레벨
8. **엘리어트** - 파동이론
9. **지지저항** - 지지선/저항선
10. **거래량** - 거래 활성도

### 4. 검색 결과 관리
- **JSON 저장**: 검색 결과를 구조화된 JSON 형태로 저장
- **메타데이터 포함**: 페이지, 청크 정보, 유사도 점수 등
- **기술적 내용 필터링**: 기술적 분석 관련 내용만 선별

## 시스템 성능

### 처리된 데이터
- **PDF 페이지**: 140페이지
- **총 텍스트**: 118,564자
- **생성된 청크**: 221개 (평균 569자)
- **기술적 내용 청크**: 157개
- **벡터 인덱스**: 221개 벡터

### 검색 성능
- **의미론적 검색**: 자연어 쿼리로 관련 내용 검색
- **키워드 매칭**: 정확한 키워드 기반 필터링
- **점수 보정**: 키워드 매칭 점수로 검색 정확도 향상

## 사용 예시

### 1. 기본 시스템 사용
```python
from DocumentsLoader.complete_faiss_system import CompleteFAISSSystem

# 시스템 초기화
system = CompleteFAISSSystem()

# 인덱스 구축 (처음 실행 시)
if system.build_index():
    print("인덱스 구축 완료")

# 의미론적 검색
results = system.semantic_search("볼린저밴드 변동성 분석")
```

### 2. 기술적 분석 검색
```python
# RSI 관련 내용 검색
rsi_results = system.technical_search("RSI")

# MACD 관련 내용 검색
macd_results = system.technical_search("MACD")

# 볼린저밴드 관련 내용 검색
bb_results = system.technical_search("볼린저밴드")
```

### 3. 검색 결과 저장
```python
# 검색 결과를 JSON 파일로 저장
system.save_search_results(results, "RSI 검색", "technical")
```

## 대화형 검색 인터페이스

시스템 실행 후 대화형 인터페이스를 통해 실시간 검색이 가능합니다:

```
🔍 대화형 검색 인터페이스
========================================
사용 가능한 명령어:
  - 'search <검색어>': 의미론적 검색
  - 'technical <지표명>': 기술적 분석 검색
  - 'stats': 인덱스 통계
  - 'quit': 종료

지원하는 지표: RSI, MACD, 볼린저밴드, 이동평균선, 스토캐스틱, 일목균형표, 피보나치, 엘리어트, 지지저항, 거래량

검색 명령어 입력: technical RSI
```

## 출력 파일

### 1. FAISS 인덱스 파일
- `DocumentsLoader/faiss_index_complete/index.faiss`: 벡터 인덱스
- `DocumentsLoader/faiss_index_complete/index.pkl`: 메타데이터

### 2. 검색 결과 파일
- `DocumentsLoader/search_results/search_results_technical_YYYYMMDD_HHMMSS.json`
- `DocumentsLoader/search_results/search_results_semantic_YYYYMMDD_HHMMSS.json`

## 시스템 특징

### 1. 한국어 최적화
- 한국어 기술적 분석 용어 인식
- 다국어 임베딩 모델 사용
- 한국어 키워드 매핑

### 2. 기술적 분석 특화
- 10개 주요 지표별 키워드 매핑
- 기술적 내용 자동 판별
- 검색 결과 필터링

### 3. 확장 가능성
- 새로운 PDF 문서 추가 가능
- 추가 기술적 분석 지표 지원
- 다양한 검색 방식 확장

## 문제 해결

### 1. 인덱스 생성 실패
```bash
# 필요한 패키지 재설치
uv add faiss-cpu sentence-transformers
```

### 2. 메모리 부족
- 청크 크기 조정 (chunk_size 파라미터)
- 배치 처리 방식 사용

### 3. 검색 결과 부족
- 키워드 매핑 확인
- 기술적 내용 필터링 조건 조정

## 성능 최적화

### 1. 인덱스 재사용
- 한 번 생성된 인덱스는 재사용 가능
- 새로운 문서 추가 시 증분 업데이트

### 2. 검색 속도
- FAISS HNSW 알고리즘으로 빠른 검색
- 벡터 인덱스 압축으로 메모리 효율성

### 3. 정확도 향상
- 키워드 매칭 점수 보정
- 기술적 내용 필터링
- 다중 키워드 검색

## 라이선스 및 저작권
- 이 시스템은 교육 및 연구 목적으로 개발되었습니다
- 실제 투자 결정 시 전문가 상담을 권장합니다
- PDF 문서의 저작권을 준수해주세요

## 문의 및 지원
프로젝트 관련 문의사항은 PROJECT_MANAGEMENT.md를 참조하거나 이슈를 등록해주세요. 