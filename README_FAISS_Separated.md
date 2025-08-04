# FAISS HNSW 시스템 - 분리된 구조

이 프로젝트는 FAISS HNSW 시스템을 **인덱스 구축**과 **검색** 두 부분으로 분리하여 구현한 교육용 시스템입니다.

## 📁 파일 구조

```
├── faiss_index_builder.py    # 인덱스 구축 및 저장
├── faiss_search_engine.py    # 검색 엔진
├── educational_faiss_system.py  # 기존 통합 시스템
└── README_FAISS_Separated.md   # 이 파일
```

## 🚀 사용법

### 1단계: 인덱스 구축

먼저 PDF 문서를 처리하여 FAISS HNSW 인덱스를 구축합니다.

```bash
# HuggingFace 임베딩으로 인덱스 구축 (기본)
python faiss_index_builder.py

# Gemini 임베딩으로 인덱스 구축
python faiss_index_builder.py gemini
```

**구축 과정:**
1. 📄 PDF 문서 로드
2. ✂️ 토크나이징 및 청킹
3. 🔢 임베딩 생성
4. 🔍 HNSW 인덱스 생성 및 저장

### 2단계: 검색 수행

구축된 인덱스를 로드하여 검색을 수행합니다.

```bash
# HuggingFace 임베딩으로 검색 (기본)
python faiss_search_engine.py

# Gemini 임베딩으로 검색
python faiss_search_engine.py gemini
```

## 🔧 주요 기능

### FAISS Index Builder (`faiss_index_builder.py`)

**주요 클래스:** `FAISSIndexBuilder`

**핵심 메서드:**
- `load_documents()`: PDF 문서 로드
- `tokenize_and_chunk()`: 토크나이징 및 청킹
- `create_embeddings()`: 임베딩 생성
- `create_hnsw_index()`: HNSW 인덱스 생성 및 저장
- `build_index()`: 전체 구축 프로세스

**특징:**
- PDF 문서를 자동으로 처리
- 기술적 분석 용어 자동 감지
- 임베딩 실패 시 재시도 로직
- 상세한 진행률 및 통계 정보

### FAISS Search Engine (`faiss_search_engine.py`)

**주요 클래스:** `FAISSSearchEngine`

**핵심 메서드:**
- `load_index()`: 저장된 인덱스 로드
- `search()`: 단일 쿼리 검색
- `batch_search()`: 배치 검색
- `interactive_search()`: 대화형 검색 모드
- `save_search_results()`: 검색 결과 저장

**특징:**
- 빠른 인덱스 로드
- 대화형 검색 인터페이스
- 검색 결과 자동 저장
- 인덱스 정보 표시

## 🎯 사용 예시

### 인덱스 구축 예시

```python
from faiss_index_builder import FAISSIndexBuilder

# 인덱스 구축기 초기화
builder = FAISSIndexBuilder(embedding_type="huggingface")

# 인덱스 구축
if builder.build_index():
    print("인덱스 구축 완료!")
```

### 검색 예시

```python
from faiss_search_engine import FAISSSearchEngine

# 검색 엔진 초기화
search_engine = FAISSSearchEngine(embedding_type="huggingface")

# 인덱스 로드
if search_engine.load_index():
    # 단일 검색
    results = search_engine.search("RSI", k=5)
    
    # 배치 검색
    queries = ["RSI", "볼린저밴드", "MACD"]
    batch_results = search_engine.batch_search(queries, k=3)
    
    # 대화형 검색
    search_engine.interactive_search()
```

## 🔍 대화형 검색 명령어

검색 엔진 실행 후 사용 가능한 명령어:

```
RSI                    # 직접 검색
볼린저밴드            # 직접 검색
search MACD           # 명시적 검색 명령
info                  # 인덱스 정보 표시
quit                  # 종료
```

## 📊 검색 결과 형식

검색 결과는 다음 정보를 포함합니다:

```json
{
  "query": "검색어",
  "embedding_type": "huggingface",
  "timestamp": "2024-01-01T12:00:00",
  "total_results": 5,
  "results": [
    {
      "score": 0.8542,
      "page": 15,
      "chunk_id": 42,
      "chunk_size": 750,
      "has_technical_content": true,
      "content": "검색된 문서 내용...",
      "metadata": {...}
    }
  ]
}
```

## ⚙️ 설정 옵션

### HNSW 설정

```python
hnsw_config = {
    'M': 16,                    # 각 노드의 최대 연결 수
    'efConstruction': 100,      # 구축 시 탐색할 이웃 수
    'efSearch': 50,             # 검색 시 탐색할 이웃 수
    'metric': faiss.METRIC_INNER_PRODUCT  # 코사인 유사도
}
```

### 텍스트 분할 설정

```python
chunk_size = 800        # 각 청크의 최대 크기
chunk_overlap = 150     # 청크 간 겹치는 부분
```

## 🔧 임베딩 모델

### HuggingFace 모델 (기본)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

### Gemini 모델
- `models/embedding-001` (Google Gemini API 필요)

## 📁 저장 위치

- **인덱스 파일:** `DocumentsLoader/educational_faiss_index/`
- **검색 결과:** `DocumentsLoader/search_results/`

## 🚨 주의사항

1. **인덱스 구축 순서:** 반드시 `faiss_index_builder.py`를 먼저 실행해야 합니다.
2. **임베딩 모델 일치:** 구축과 검색 시 동일한 임베딩 모델을 사용해야 합니다.
3. **API 키 설정:** Gemini 사용 시 `GOOGLE_API_KEY` 환경 변수 설정이 필요합니다.
4. **PDF 파일 경로:** `DocumentsLoader/data/기술적차트분석이론및방법.pdf` 파일이 있어야 합니다.
5. **보안 설정:** `allow_dangerous_deserialization=True`가 설정되어 있습니다. 이는 로컬에서 생성한 파일만 로드하므로 안전합니다.

## 🔄 기존 시스템과의 차이점

| 기능 | 통합 시스템 | 분리된 시스템 |
|------|-------------|---------------|
| 인덱스 구축 | ✅ | ✅ (별도 파일) |
| 검색 | ✅ | ✅ (별도 파일) |
| 대화형 모드 | ✅ | ✅ |
| 배치 검색 | ❌ | ✅ |
| 모듈화 | ❌ | ✅ |
| 재사용성 | 낮음 | 높음 |

## 💡 활용 팁

1. **대용량 문서 처리:** 인덱스 구축은 한 번만 수행하고, 검색은 여러 번 수행
2. **배치 검색:** 여러 쿼리를 한 번에 처리하여 효율성 향상
3. **결과 저장:** 검색 결과를 JSON으로 저장하여 나중에 분석 가능
4. **모델 변경:** 구축과 검색 시 동일한 임베딩 모델 사용 필수

## 🐛 문제 해결

### 인덱스 파일을 찾을 수 없음
```bash
# 먼저 인덱스 구축
python faiss_index_builder.py
```

### Pickle 보안 오류
```
❌ 인덱스 로드 실패: The de-serialization relies loading a pickle file...
```
**해결방법:** 코드에서 `allow_dangerous_deserialization=True`가 이미 설정되어 있습니다. 이는 로컬에서 생성한 파일만 로드하므로 안전합니다.

### Gemini API 오류
```bash
# HuggingFace 모델 사용
python faiss_index_builder.py huggingface
python faiss_search_engine.py huggingface
```

### 메모리 부족
- 청크 크기를 줄이거나
- 더 작은 임베딩 모델 사용

---

**🎓 교육 목적:** 이 시스템은 FAISS HNSW의 각 단계를 명확히 구분하여 RAG 시스템의 전체 과정을 학습할 수 있도록 설계되었습니다. 