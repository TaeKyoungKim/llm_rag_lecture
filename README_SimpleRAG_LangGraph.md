# SimpleRAG with LangGraph

LangGraph를 사용하여 구현한 SimpleRAG 시스템입니다. `FAISSSearchEngine` 클래스를 임포트하여 검색과 답변 생성을 분리된 노드로 구성했습니다.

## 🏗️ 시스템 아키텍처

### LangGraph 워크플로우
```
입력 쿼리 → [검색 노드] → [조건부 분기] → [답변 생성 노드] → 최종 답변
                    ↓
              [오류 처리 노드]
```

### 노드 구성
1. **검색 노드 (search)**: FAISS 검색 수행
2. **답변 생성 노드 (generate_answer)**: LLM으로 답변 생성
3. **오류 처리 노드 (handle_error)**: 오류 상황 처리

## 🚀 실행 방법

### 1. 기본 실행
```bash
# 기본 설정 (HuggingFace 임베딩 + simple 프롬프트)
uv run python simple_rag_langgraph.py

# Gemini 임베딩 사용
uv run python simple_rag_langgraph.py gemini

# Gemini 임베딩 + detailed 프롬프트
uv run python simple_rag_langgraph.py gemini detailed

# Gemini 임베딩 + academic 프롬프트
uv run python simple_rag_langgraph.py gemini academic
```

### 2. 사용 가능한 프롬프트 스타일

| 스타일 | 특징 | 사용 시기 |
|--------|------|-----------|
| **simple** | 간결하고 핵심적인 답변 | 빠른 참조 및 요약 |
| **detailed** | 상세하고 구조화된 답변 | 일반적인 학습 및 이해 |
| **academic** | 학술적이고 체계적인 답변 | 연구 및 전문적 분석 |

## 📁 파일 구조

```
llm_rag_lecture/
├── simple_rag_langgraph.py          # 메인 SimpleRAG 시스템
├── faiss_search_engine.py           # FAISS 검색 엔진 (임포트됨)
├── faiss_index_builder.py           # 인덱스 구축 도구
├── DocumentsLoader/
│   ├── educational_faiss_index/     # FAISS 인덱스 저장소
│   └── simple_rag_results/          # SimpleRAG 결과 저장소
└── README_SimpleRAG_LangGraph.md    # 이 파일
```

## 🔧 주요 기능

### 1. LangGraph 워크플로우
- **상태 관리**: `RAGState` TypedDict로 상태 정의
- **노드 분리**: 검색과 답변 생성을 독립적인 노드로 구성
- **조건부 분기**: 검색 성공/실패에 따른 분기 처리
- **오류 처리**: 전용 오류 처리 노드

### 2. FAISS 검색 엔진 통합
- `FAISSSearchEngine` 클래스 임포트
- HNSW 인덱스 활용
- 유사도 검색 및 결과 정렬

### 3. LLM 통합
- Gemini 2.5 Flash 모델 사용
- 다양한 프롬프트 스타일 지원
- ChatPromptTemplate 활용

### 4. 결과 저장
- JSON 형태로 결과 저장
- 메타데이터 포함 (임베딩 타입, LLM 모델, 프롬프트 스타일 등)
- 타임스탬프 자동 생성

## 💻 사용 예시

### 대화형 모드
```bash
$ uv run python simple_rag_langgraph.py gemini simple

🤖 SimpleRAG with LangGraph
   • 임베딩 모델: gemini
   • 프롬프트 스타일: simple
   • 워크플로우 엔진: LangGraph

🤖 SimpleRAG 대화형 모드
============================================================
사용 가능한 명령어:
  - 질문만 입력: 직접 질문 (예: 'RSI란 무엇인가요?', '볼린저밴드 사용법')
  - 'ask <질문>': 명시적 질문 명령
  - 'info': 시스템 정보
  - 'quit': 종료

현재 설정:
  • 임베딩: gemini
  • LLM: Gemini 2.5 Flash
  • 프롬프트 스타일: simple
  • 워크플로우: LangGraph

질문 입력: RSI란 무엇인가요?
```

### 워크플로우 실행 과정
```
🔍 검색 노드 실행: 'RSI란 무엇인가요?'
🤖 답변 생성 노드 실행

🤖 답변:
----------------------------------------
RSI는 상대강도지수(Relative Strength Index)로...

📄 참고 문서 (3개):
   1. 페이지 15 (유사도: 0.8923)
   2. 페이지 23 (유사도: 0.7845)
   3. 페이지 31 (유사도: 0.7234)
```

## 🔍 LangGraph vs 기존 RAG 시스템

### LangGraph의 장점
1. **모듈화**: 검색과 답변 생성을 독립적인 노드로 분리
2. **확장성**: 새로운 노드나 조건부 분기 쉽게 추가 가능
3. **상태 관리**: TypedDict로 명확한 상태 정의
4. **오류 처리**: 전용 오류 처리 노드로 견고한 시스템
5. **시각화**: 워크플로우 구조를 명확하게 파악 가능

### 기존 RAG 시스템과의 차이점
| 구분 | 기존 RAG | LangGraph RAG |
|------|----------|---------------|
| 구조 | 단일 함수 | 노드 기반 워크플로우 |
| 확장성 | 제한적 | 높음 |
| 오류 처리 | try-catch | 전용 노드 |
| 상태 관리 | 변수 | TypedDict |
| 디버깅 | 어려움 | 쉬움 |

## 🛠️ 시스템 설정

### 환경 변수
```bash
# .env 파일
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 의존성 패키지
```toml
# pyproject.toml
langgraph = "^0.2.0"
langchain = "^0.3.0"
langchain-google-genai = "^2.0.0"
langchain-community = "^0.3.0"
faiss-cpu = "^1.7.4"
sentence-transformers = "^3.0.0"
python-dotenv = "^1.0.0"
```

## 📊 성능 최적화

### 1. 검색 최적화
- HNSW efSearch 파라미터 조정
- 검색 결과 수 (k) 최적화
- 임베딩 모델 선택

### 2. LLM 최적화
- 프롬프트 스타일 선택
- temperature 조정
- max_output_tokens 설정

### 3. 워크플로우 최적화
- 노드 간 데이터 전달 최적화
- 불필요한 상태 업데이트 제거
- 메모리 사용량 모니터링

## 🔧 문제 해결

### 1. LangGraph 관련 오류
```bash
# LangGraph 설치 확인
uv add langgraph

# 버전 호환성 확인
uv list | grep langgraph
```

### 2. FAISS 인덱스 로드 오류
```bash
# 인덱스 파일 존재 확인
ls DocumentsLoader/educational_faiss_index/

# 인덱스 재구축
uv run python faiss_index_builder.py
```

### 3. API 키 오류
```bash
# 환경 변수 확인
echo $GOOGLE_API_KEY

# .env 파일 확인
cat .env
```

## 🎯 향후 개선 방향

1. **다중 검색 노드**: 다양한 검색 방법 통합
2. **답변 검증 노드**: 생성된 답변의 품질 검증
3. **사용자 피드백 노드**: 사용자 피드백 수집 및 학습
4. **캐싱 노드**: 자주 묻는 질문 캐싱
5. **로깅 노드**: 상세한 실행 로그 수집

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

---

**참고**: 이 시스템을 사용하기 전에 `faiss_index_builder.py`를 실행하여 FAISS 인덱스를 먼저 구축해야 합니다. 