# FAISS RAG 시스템

FAISS 검색 결과를 Gemini 2.5 Flash 모델로 답변을 생성하는 완전한 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 🚀 주요 기능

### 1. **FAISS HNSW 벡터 검색**
- 고성능 HNSW(Hierarchical Navigable Small World) 알고리즘 사용
- 빠른 유사도 검색 및 정확한 문서 검색
- 코사인 유사도 기반 검색

### 2. **Gemini 2.5 Flash LLM**
- Google의 최신 Gemini 2.5 Flash 모델 사용
- 빠른 응답 속도와 높은 품질의 답변 생성
- 한국어 지원

### 3. **완전한 RAG 파이프라인**
- **1단계**: 질문을 임베딩으로 변환
- **2단계**: FAISS에서 유사한 문서 검색
- **3단계**: 검색된 문서를 컨텍스트로 구성
- **4단계**: Gemini 2.5 Flash로 답변 생성

### 4. **대화형 인터페이스**
- 직관적인 대화형 질문-답변 모드
- 실시간 검색 결과 및 답변 생성
- 자동 결과 저장

## 📁 파일 구조

```
llm_rag_lecture/
├── faiss_rag_system.py              # 메인 RAG 시스템
├── faiss_index_builder.py           # FAISS 인덱스 구축
├── faiss_search_engine.py           # 검색 전용 엔진
├── DocumentsLoader/
│   ├── educational_faiss_index/     # FAISS 인덱스 저장소
│   ├── rag_results/                 # RAG 결과 저장소
│   └── search_results/              # 검색 결과 저장소
└── README_FAISS_RAG_System.md       # 이 파일
```

## 🛠️ 설치 및 설정

### 1. 환경 설정
```bash
# .env 파일 생성
echo "GOOGLE_API_KEY=your_gemini_api_key_here" > .env
```

### 2. 의존성 설치
```bash
# uv를 사용한 설치
uv sync
```

### 3. FAISS 인덱스 구축 (최초 1회)
```bash
# Gemini 임베딩으로 인덱스 구축
uv run python faiss_index_builder.py gemini

# 또는 HuggingFace 임베딩으로 구축
uv run python faiss_index_builder.py huggingface
```

## 🚀 사용법

### 1. RAG 시스템 실행
```bash
# Gemini 임베딩 + Gemini 2.5 Flash LLM
uv run python faiss_rag_system.py gemini

# HuggingFace 임베딩 + Gemini 2.5 Flash LLM
uv run python faiss_rag_system.py huggingface
```

### 2. 대화형 모드 사용
```
🤖 대화형 RAG 모드
============================================================
사용 가능한 명령어:
  - 질문만 입력: 직접 질문 (예: 'RSI란 무엇인가요?', '볼린저밴드 사용법')
  - 'ask <질문>': 명시적 질문 명령
  - 'info': 시스템 정보
  - 'quit': 종료

현재 모델:
  • 임베딩: gemini
  • LLM: Gemini 2.5 Flash

질문 입력: RSI란 무엇인가요?
```

### 3. 질문 예시
```
질문 입력: RSI란 무엇인가요?
질문 입력: 볼린저밴드 사용법을 알려주세요
질문 입력: MACD 지표의 의미는?
질문 입력: 기술적 분석의 기본 원리는?
질문 입력: 엘리어트 파동이론이란?
```

## 🔧 시스템 구성

### 임베딩 모델 옵션
1. **Gemini Embeddings** (`models/embedding-001`)
   - Google의 최신 임베딩 모델
   - 높은 품질의 벡터 생성
   - API 키 필요

2. **HuggingFace Embeddings**
   - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
   - `sentence-transformers/all-MiniLM-L6-v2`
   - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
   - 오프라인 사용 가능

### LLM 모델
- **Gemini 2.5 Flash** (`gemini-2.0-flash-exp`)
  - 빠른 응답 속도
  - 높은 품질의 답변
  - 한국어 지원

### FAISS 설정
```python
hnsw_config = {
    'M': 16,                    # 각 노드의 최대 연결 수
    'efConstruction': 100,      # 구축 시 탐색할 이웃 수
    'efSearch': 50,             # 검색 시 탐색할 이웃 수
    'metric': faiss.METRIC_INNER_PRODUCT  # 코사인 유사도
}
```

## 📊 출력 예시

### 검색 과정
```
🔍 질문: '볼린저밴드란?'
============================================================
   🔧 HNSW efSearch 설정: 50
   🔄 1단계: 유사도 검색 수행 중...
   ✅ 5개 관련 문서 발견
   🔄 2단계: 컨텍스트 구성 중...

📄 검색된 문서 (5개):
----------------------------------------
   📄 문서 1:
      - 유사도 점수: 1.2606
      - 페이지: 122
      - 청크 크기: 774자
      - 기술적 내용: ❌
      - 내용 미리보기: ◆포트폴리오이론...

   ✅ 컨텍스트 구성 완료 (2472자)
   🔄 3단계: Gemini 2.5 Flash로 답변 생성 중...
   ✅ 답변 생성 완료
```

### 답변 출력
```
🤖 답변:
----------------------------------------
문서 2에 따르면, 볼린저 밴드는 주가 변동성의 특성(시변성, 운집성, 평균 회귀성)을 반영하여 만들어진 지표입니다.

*   **시변성:** 주가의 변동성은 시간에 따라 변하며, 볼린저 밴드의 두께도 이에 따라 변합니다...
*   **운집성:** 변동성이 커지면 바로 기존의 변동성으로 돌아가지 않고, 한동안 커진 상태를 유지합니다...
*   **평균 회귀성:** 변동성이 커지더라도 결국에는 평균적인 변동성으로 돌아옵니다...

요약하자면, 볼린저 밴드는 주가 변동성의 변화를 보여주는 지표이며, 수축과 확장을 반복하면서 시장의 변동성을 시각적으로 나타냅니다.
```

## 💾 결과 저장

### RAG 결과 저장
- 위치: `DocumentsLoader/rag_results/`
- 형식: JSON
- 포함 정보:
  - 질문과 답변
  - 사용된 모델 정보
  - 검색된 문서 소스
  - 타임스탬프

### 저장 예시
```json
{
  "query": "볼린저밴드란?",
  "answer": "문서 2에 따르면, 볼린저 밴드는 주가 변동성의 특성...",
  "embedding_type": "gemini",
  "llm_model": "gemini-2.5-flash",
  "timestamp": "2024-01-15T10:30:45.123456",
  "sources": [
    {
      "page": "32",
      "chunk_id": "chunk_001",
      "score": 1.2717,
      "content_preview": "볼린저밴드의성질을이해하기위해서는..."
    }
  ],
  "context_preview": "[문서 1] ◆포트폴리오이론...\n\n[문서 2] 볼린저밴드의성질..."
}
```

## 🔍 문제 해결

### 1. API 키 오류
```
❌ LLM 모델 로드 실패: GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.
```
**해결책**: `.env` 파일에 올바른 Gemini API 키를 설정하세요.

### 2. 인덱스 파일 없음
```
❌ FAISS 인덱스 파일을 찾을 수 없습니다
```
**해결책**: 먼저 `faiss_index_builder.py`를 실행하여 인덱스를 구축하세요.

### 3. 네트워크 오류
```
❌ API 할당량 초과입니다
```
**해결책**: 잠시 후 다시 시도하거나 다른 API 키를 사용하세요.

### 4. 임베딩 모델 로드 실패
```
❌ HuggingFace 임베딩 로드 실패
```
**해결책**: 인터넷 연결을 확인하고 필요한 모델을 다운로드하세요.

## 🎯 성능 최적화

### 1. 검색 성능
- HNSW 파라미터 조정으로 검색 속도와 정확도 균형
- `efSearch` 값을 높이면 정확도 향상, 속도 감소
- `efSearch` 값을 낮추면 속도 향상, 정확도 감소

### 2. 답변 품질
- 검색할 문서 수(`k`) 조정
- 프롬프트 템플릿 최적화
- LLM 파라미터(temperature, max_output_tokens) 조정

### 3. 메모리 사용량
- 컨텍스트 길이 제한
- 배치 처리 구현
- 불필요한 데이터 제거

## 🔄 업데이트 및 확장

### 새로운 문서 추가
1. `faiss_index_builder.py`에서 PDF 경로 수정
2. 인덱스 재구축 실행
3. 새로운 문서로 RAG 시스템 업데이트

### 다른 LLM 모델 사용
1. `_initialize_llm()` 메서드에서 모델 변경
2. 프롬프트 템플릿 조정
3. 출력 파서 수정

### 웹 인터페이스 추가
- Streamlit 또는 Gradio를 사용한 웹 UI
- REST API 구현
- 실시간 채팅 인터페이스

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

## 🤝 기여

버그 리포트, 기능 요청, 개선 사항은 언제든 환영합니다!

---

**FAISS RAG 시스템**으로 기술적 분석 문서에 대한 지능형 질의응답을 경험해보세요! 🚀 