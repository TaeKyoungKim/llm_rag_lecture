# 🎓 교육용 FAISS HNSW 시스템

이 프로젝트는 RAG (Retrieval-Augmented Generation) 시스템의 핵심 구성 요소인 FAISS HNSW를 단계별로 학습할 수 있도록 구현된 교육용 코드입니다.

## 📚 학습 목표

이 시스템을 통해 다음을 학습할 수 있습니다:

1. **문서 로딩**: PDF 파일을 읽고 처리하는 방법
2. **토크나이징**: 텍스트를 적절한 크기로 분할하는 방법
3. **임베딩**: 텍스트를 벡터로 변환하는 방법
4. **HNSW 인덱싱**: FAISS HNSW를 사용한 벡터 인덱싱
5. **유사도 검색**: 쿼리와 유사한 문서를 찾는 방법

## 🏗️ 시스템 아키텍처

```
📄 PDF 문서
    ↓ (1단계: 문서 로드)
📄 LangChain Documents
    ↓ (2단계: 토크나이징/청킹)
✂️ 텍스트 청크들
    ↓ (3단계: 임베딩)
🔢 벡터들
    ↓ (4단계: HNSW 인덱싱)
🔍 FAISS HNSW 인덱스
    ↓ (5단계: 검색)
🎯 유사도 검색 결과
```

## 🚀 설치 및 실행

### 1. 의존성 설치

```bash
pip install faiss-cpu langchain langchain-community langchain-google-genai sentence-transformers PyPDF2 numpy
```

### 2. 환경 변수 설정 (Gemini 사용 시)

```bash
export GOOGLE_API_KEY="your_google_api_key_here"
```

### 3. 실행

```bash
# HuggingFace 임베딩 사용 (기본)
python educational_faiss_system.py

# Gemini 임베딩 사용
python educational_faiss_system.py gemini
```

## 📖 단계별 상세 설명

### 1단계: 시스템 초기화

```python
# 임베딩 모델 초기화
- HuggingFace: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Gemini: models/embedding-001

# 텍스트 분할기 설정
- chunk_size: 800 (각 청크의 최대 크기)
- chunk_overlap: 150 (청크 간 겹치는 부분)

# HNSW 설정
- M: 16 (각 노드의 최대 연결 수)
- efConstruction: 100 (구축 시 탐색할 이웃 수)
- efSearch: 50 (검색 시 탐색할 이웃 수)
```

### 2단계: PDF 문서 로드

```python
def step1_load_documents(self) -> List[Document]:
    """
    PDF 파일을 읽어서 LangChain Document 형식으로 변환
    - 각 페이지를 개별 Document로 변환
    - 메타데이터 추가 (페이지 번호, 파일 경로 등)
    """
```

**학습 포인트:**
- PDF 파일 처리 방법
- LangChain Document 구조
- 메타데이터 관리

### 3단계: 토크나이징 및 청킹

```python
def step2_tokenize_and_chunk(self, documents: List[Document]) -> List[Document]:
    """
    문서를 작은 청크로 분할하여 처리 효율성 향상
    - RecursiveCharacterTextSplitter 사용
    - 청크별 메타데이터 추가
    - 기술적 분석 내용 포함 여부 확인
    """
```

**학습 포인트:**
- 텍스트 분할 전략
- 청크 크기 최적화
- 컨텍스트 유지를 위한 오버랩 설정

### 4단계: 임베딩 생성

```python
def step3_create_embeddings(self, documents: List[Document]) -> List[np.ndarray]:
    """
    각 문서 청크를 벡터로 변환
    - HuggingFace 또는 Gemini 임베딩 모델 사용
    - 진행률 표시
    - 임베딩 차원 확인
    """
```

**학습 포인트:**
- 임베딩 모델 선택
- 벡터 차원 이해
- 임베딩 품질 평가

### 5단계: HNSW 인덱스 생성

```python
def step4_create_hnsw_index(self, documents: List[Document], embeddings: List[np.ndarray]) -> bool:
    """
    FAISS HNSW 인덱스를 생성하고 벡터를 저장
    - HNSW 파라미터 설정
    - 벡터를 인덱스에 추가
    - LangChain FAISS 래퍼 생성
    """
```

**학습 포인트:**
- HNSW 알고리즘 이해
- FAISS 인덱스 구조
- 벡터 저장 및 인덱싱

### 6단계: 유사도 검색

```python
def step5_similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
    """
    쿼리와 유사한 문서를 HNSW 인덱스에서 검색
    - HNSW 검색 파라미터 설정
    - 유사도 점수 계산
    - 결과 정렬 및 반환
    """
```

**학습 포인트:**
- 유사도 계산 방법
- 검색 결과 평가
- 검색 성능 최적화

## 🎯 사용 예시

### 대화형 검색

```bash
검색 명령어 입력: search RSI 지표 활용법
검색 명령어 입력: search 볼린저 밴드 분석
검색 명령어 입력: stats
검색 명령어 입력: quit
```

### 프로그래밍 방식 사용

```python
from educational_faiss_system import EducationalFAISSSystem

# 시스템 초기화
system = EducationalFAISSSystem(embedding_type="huggingface")

# 전체 시스템 구축
if system.build_complete_system():
    # 검색 수행
    results = system.step5_similarity_search("RSI 지표", k=3)
    
    # 결과 출력
    for doc, score in results:
        print(f"유사도: {score:.4f}")
        print(f"내용: {doc.page_content[:100]}...")
```

## 📊 시스템 통계

실행 후 다음과 같은 통계 정보를 확인할 수 있습니다:

```
📊 시스템 통계:
   • 임베딩 모델: huggingface
   • 인덱스 타입: FAISS HNSW
   • 저장 위치: DocumentsLoader/educational_faiss_index
   • HNSW 노드 수: 1234
   • HNSW 최대 레벨: 5
   • HNSW efSearch: 50
   • 총 문서 수: 156
   • 기술적 내용 문서: 89
   • 평균 문서 길이: 756자
```

## 🔧 주요 설정 파라미터

### 임베딩 모델
- **HuggingFace**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Gemini**: `models/embedding-001`

### 텍스트 분할
- **chunk_size**: 800 (각 청크의 최대 크기)
- **chunk_overlap**: 150 (청크 간 겹치는 부분)
- **separators**: `["\n\n", "\n", " ", ""]` (분할 기준 구분자)

### HNSW 설정
- **M**: 16 (각 노드의 최대 연결 수)
- **efConstruction**: 100 (구축 시 탐색할 이웃 수)
- **efSearch**: 50 (검색 시 탐색할 이웃 수)
- **metric**: `faiss.METRIC_INNER_PRODUCT` (코사인 유사도)

## 📁 파일 구조

```
llm_rag_lecture/
├── educational_faiss_system.py          # 메인 교육용 시스템
├── README_Educational_FAISS.md         # 이 파일
├── DocumentsLoader/
│   ├── data/
│   │   └── 기술적차트분석이론및방법.pdf  # 입력 PDF 파일
│   ├── educational_faiss_index/        # 생성된 인덱스 저장 위치
│   └── search_results/                 # 검색 결과 저장 위치
```

## 🎓 학습 체크리스트

- [ ] PDF 파일 로딩 과정 이해
- [ ] 텍스트 분할 전략 이해
- [ ] 임베딩 모델의 역할 이해
- [ ] HNSW 알고리즘의 기본 개념 이해
- [ ] FAISS 인덱스 구조 이해
- [ ] 유사도 검색 과정 이해
- [ ] 검색 결과 평가 방법 이해

## 🔍 추가 학습 자료

1. **FAISS 공식 문서**: https://github.com/facebookresearch/faiss
2. **HNSW 논문**: https://arxiv.org/abs/1603.09320
3. **LangChain 문서**: https://python.langchain.com/
4. **RAG 시스템 개요**: https://arxiv.org/abs/2005.11401

## 🐛 문제 해결

### 일반적인 오류

1. **PDF 파일을 찾을 수 없음**
   - `DocumentsLoader/data/` 폴더에 PDF 파일이 있는지 확인

2. **Gemini API 오류**
   ```bash
   # API 키 설정 도우미 실행
   python setup_gemini_api.py
   
   # 또는 수동으로 환경 변수 설정
   export GOOGLE_API_KEY="your_api_key_here"
   ```

3. **임베딩 모델 로드 실패**
   - 인터넷 연결 확인
   - 모델 이름 확인
   - 시스템이 자동으로 대안 모델을 시도합니다

4. **메모리 부족**
   - 청크 크기 줄이기
   - 배치 크기 조정

### Gemini API 문제 해결

#### API 키 설정
```bash
# 1. Google AI Studio에서 API 키 생성
# https://makersuite.google.com/app/apikey

# 2. 환경 변수 설정
export GOOGLE_API_KEY="your_api_key_here"

# 3. 연결 테스트
python setup_gemini_api.py
```

#### 일반적인 API 오류
- **invalid_grant/Bad Request**: API 키 인증 오류
- **timeout**: 네트워크 타임아웃
- **quota exceeded**: API 할당량 초과

#### 해결 방법
1. **HuggingFace 임베딩 사용** (권장)
   ```bash
   python educational_faiss_system.py
   ```

2. **API 키 재설정**
   ```bash
   python setup_gemini_api.py
   ```

3. **환경 변수 확인**
   ```bash
   echo $GOOGLE_API_KEY
   ```

### 성능 최적화

1. **검색 속도 향상**
   - `efSearch` 값 조정
   - HNSW 파라미터 최적화

2. **검색 정확도 향상**
   - 임베딩 모델 변경
   - 청크 크기 조정

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다. 자유롭게 사용하고 수정하실 수 있습니다. 