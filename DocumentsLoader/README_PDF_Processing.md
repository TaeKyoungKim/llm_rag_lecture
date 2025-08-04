# PDF 처리 시스템 사용 가이드

## 개요
이 시스템은 주식 기술적 분석 PDF 문서를 자동으로 로드하고 분석하여 핵심 정보를 추출하는 도구입니다.

## 파일 구조
```
DocumentsLoader/
├── data/
│   └── 기술적차트분석이론및방법.pdf  # 분석 대상 PDF
├── PDFLoader_stock.py                    # 기존 시연 코드
├── process_technical_analysis_pdf.py     # 기본 PDF 처리 시스템
├── process_technical_analysis_pdf_improved.py  # 개선된 PDF 처리 시스템
└── README_PDF_Processing.md              # 이 파일
```

## 설치 및 실행

### 1. 필요한 패키지 설치
```bash
uv add langchain_community pypdf pandas requests
```

### 2. 기본 PDF 처리 시스템 실행
```bash
uv run DocumentsLoader/process_technical_analysis_pdf.py
```

### 3. 개선된 PDF 처리 시스템 실행
```bash
uv run DocumentsLoader/process_technical_analysis_pdf_improved.py
```

## 주요 기능

### 1. PDF 텍스트 추출
- PyPDFLoader를 사용한 140페이지 PDF 처리
- 총 118,564자 텍스트 추출
- 메타데이터 자동 수집

### 2. 기술적 분석 지표 인식
자동으로 다음 지표들을 인식하고 분류합니다:

#### 기술 지표 (14개 발견)
- **RSI** (Relative Strength Index, 상대강도지수)
- **MACD** (Moving Average Convergence & Divergence)
- **볼린저밴드** (Bollinger Bands)
- **이동평균선** (Moving Average)
- **스토캐스틱** (Stochastic)
- **일목균형표** (Ichimoku Kinko Hyo)
- **피보나치되돌림** (Fibonacci Retracement)
- **엘리어트파동** (Elliott Wave Theory)

#### 핵심 개념 (12개 발견)
- **지지선/저항선** (Support/Resistance)
- **추세선** (Trend Line)
- **과매수/과매도** (Overbought/Oversold)
- **다이버전스** (Divergence)
- **골든크로스/데드크로스** (Golden Cross/Death Cross)
- **거래량** (Volume)
- **매물대** (Supply Zone)

### 3. 품질 평가 시스템
- **100/100점**: 최우수 (전문 수준)
- **80-99점**: 우수 (실용 수준)
- **60-79점**: 양호 (기초 수준)
- **40-59점**: 기본 (입문 수준)

### 4. 자동 리포트 생성
- PDF 기본 정보 (파일명, 페이지 수, 텍스트 길이)
- 발견된 기술 지표 및 개념 목록
- 주요 포커스 분석
- 품질 점수 및 등급
- 전체 내용 텍스트

## 출력 파일

### 1. technical_analysis_results.txt
기본 PDF 처리 시스템의 결과

### 2. improved_technical_analysis_results.txt
개선된 PDF 처리 시스템의 결과 (권장)

## 시스템 개선 사항

### 버전 1 (기본)
- 기본적인 키워드 매칭
- 단순한 텍스트 추출

### 버전 2 (개선)
- 정규표현식을 활용한 정확한 패턴 매칭
- 한국어 기술적 분석 용어 최적화
- 더 포괄적인 키워드 데이터베이스
- 향상된 품질 평가 알고리즘

## 사용 예시

### 1. 기본 실행
```python
from DocumentsLoader.process_technical_analysis_pdf import RealPDFProcessor

processor = RealPDFProcessor()
result = processor.process_pdf()
```

### 2. 개선된 버전 실행
```python
from DocumentsLoader.process_technical_analysis_pdf_improved import ImprovedPDFProcessor

processor = ImprovedPDFProcessor()
result = processor.process_pdf()
```

## 분석 결과 예시

```
📈 기술적 분석 PDF 상세 리포트
============================================================

📄 PDF 기본 정보:
   • 파일명: 기술적차트분석이론및방법.pdf
   • 문서 수: 140개
   • 총 텍스트 길이: 118,564자

🔍 기술적 분석 내용 추출 결과:
   • 발견된 기술 지표: 14개
   • 발견된 패턴: 0개
   • 발견된 개념: 12개
   • 주요 포커스: RSI + MACD 복합 분석
   • 내용 품질 점수: 100/100
   • 품질 등급: 🥇 최우수 (전문 수준)
```

## 확장 가능한 기능

### 1. RAG 시스템 통합
- 추출된 내용을 벡터 데이터베이스에 저장
- LLM과 연동한 질의응답 시스템

### 2. 실시간 분석
- 실시간 주식 데이터와 연동
- 동적 기술적 분석 지표 계산

### 3. 웹 인터페이스
- 사용자 친화적인 웹 기반 도구
- 시각적 차트 및 분석 결과 표시

### 4. 다국어 지원
- 영어, 일본어 등 다양한 언어의 기술적 분석 문서 처리
- 자동 번역 및 용어 매핑

## 문제 해결

### 1. PDF 로드 실패
```bash
# pypdf 패키지 설치 확인
uv add pypdf
```

### 2. 인코딩 문제
- 모든 파일은 UTF-8 인코딩으로 저장
- 한국어 텍스트 처리 최적화

### 3. 메모리 부족
- 대용량 PDF의 경우 청크 단위 처리
- 스트리밍 방식의 텍스트 추출

## 라이선스 및 저작권
- 이 시스템은 교육 및 연구 목적으로 개발되었습니다
- 실제 투자 결정 시 전문가 상담을 권장합니다
- PDF 문서의 저작권을 준수해주세요

## 문의 및 지원
프로젝트 관련 문의사항은 PROJECT_MANAGEMENT.md를 참조하거나 이슈를 등록해주세요. 