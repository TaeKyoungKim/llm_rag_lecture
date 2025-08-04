# LangChain WebBaseLoader - 주식 기술적 분석 사이트 통합 샘플 코드 (수정됨)

# 필요한 패키지 설치
# pip install -qU langchain_community beautifulsoup4 nest_asyncio pandas

import asyncio
import nest_asyncio
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader
from datetime import datetime
import re
import warnings

# Jupyter 환경에서 asyncio 사용 시 필요
nest_asyncio.apply()

# 경고 메시지 필터링
warnings.filterwarnings('ignore', category=DeprecationWarning)

# === 주식 기술적 분석 관련 사이트 URL 모음 ===
TECHNICAL_ANALYSIS_SITES = {
    "global": [
        "https://stockcharts.com/",
        "https://www.tradingview.com/",
        "https://www.investing.com/technical/",
        "https://www.prorealtime.com/en/",
        "https://zerodha.com/varsity/module/technical-analysis/",
        "https://www.cmcmarkets.com/en-gb/trading-guides/technical-indicators"
    ],
    "korean": [
        "https://kr.tradingview.com/",
        "https://kr.investing.com/technical/indices-indicators",
        "https://www.paxnet.co.kr/",
        "http://data.krx.co.kr/",
        "https://www.fnguide.com/"
    ],
    "blogs_and_guides": [
        "https://blog.naver.com/kihyun113/222898232520",  # RSI 지표 활용법
        "https://blog.naver.com/parkjongpir/222234341982",  # 보조지표 정리
        "https://contents.premium.naver.com/yonseident/ysdent/contents/241109181740264ic",  # S+R+M 조합
        "https://blog.okfngroup.com/content/how-to-read-the-rsi-indicator"  # RSI 지표 보는 법
    ]
}

# === 1. 기본 기술적 분석 사이트 스크래핑 ===
print("=== 1. 기본 기술적 분석 사이트 스크래핑 ===")

def load_technical_analysis_site(url, description=""):
    """기술적 분석 사이트를 안전하게 로드하는 함수"""
    try:
        loader = WebBaseLoader(url)
        loader.requests_kwargs = {
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            },
            'timeout': 10,
            'verify': True
        }
        docs = loader.load()
        print(f"✅ {description} ({url}): 로드 성공 - {len(docs)}개 문서, {len(docs[0].page_content)}자")
        return docs
    except Exception as e:
        print(f"❌ {description} ({url}): 로드 실패 - {str(e)}")
        return None

# 주요 기술적 분석 사이트 테스트
sample_sites = [
    ("https://www.example.com/", "예제 사이트"),
    ("https://httpbin.org/html", "HTML 테스트 사이트"),
]

for url, description in sample_sites:
    docs = load_technical_analysis_site(url, description)
    if docs:
        print(f"  내용 미리보기: {docs[0].page_content[:100]}...")
print()

# === 2. 주식 기술적 분석 정보 수집 시스템 (수정됨) ===
print("=== 2. 주식 기술적 분석 정보 수집 시스템 ===")

class TechnicalAnalysisCollector:
    """기술적 분석 정보를 수집하는 클래스"""
    
    def __init__(self):
        self.collected_data = []
        self.failed_urls = []
    
    def extract_technical_keywords(self, content):
        """기술적 분석 관련 키워드 추출"""
        keywords = {
            'indicators': ['RSI', 'MACD', '볼린저밴드', '스토캐스틱', '이동평균', 'ATR', 'CCI'],
            'patterns': ['삼각형', '쐐기형', '플래그', '헤드앤숄더', '더블탑', '더블바텀'],
            'concepts': ['지지선', '저항선', '추세선', '과매수', '과매도', '다이버전스', '크로스']
        }
        
        found_keywords = {}
        content_lower = content.lower()
        
        for category, word_list in keywords.items():
            found_keywords[category] = []
            for keyword in word_list:
                if keyword.lower() in content_lower or keyword in content:
                    found_keywords[category].append(keyword)
        
        return found_keywords
    
    def analyze_content(self, docs, source_url):
        """문서 내용 분석"""
        if not docs:
            return None
            
        doc = docs[0]
        content = doc.page_content
        
        analysis = {
            'source': source_url,
            'title': doc.metadata.get('title', 'N/A'),
            'content_length': len(content),
            'keywords': self.extract_technical_keywords(content),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return analysis
    
    async def collect_single_site(self, url):
        """단일 사이트에서 정보 수집 (수정된 비동기 방법)"""
        try:
            loader = WebBaseLoader(url)
            loader.requests_kwargs = {
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                },
                'timeout': 15
            }
            loader.requests_per_second = 1  # 서버 부하 방지
            
            # aload()는 비동기 제너레이터이므로 async for로 처리
            docs = []
            async for doc in loader.alazy_load():
                docs.append(doc)
            
            analysis = self.analyze_content(docs, url)
            
            if analysis:
                self.collected_data.append(analysis)
                print(f"✅ {url}: 수집 완료")
                return analysis
            else:
                print(f"❌ {url}: 분석 실패")
                return None
                
        except Exception as e:
            self.failed_urls.append((url, str(e)))
            print(f"❌ {url}: 수집 실패 - {str(e)[:50]}...")
            return None
    
    def collect_from_sites(self, urls, max_concurrent=3):
        """여러 사이트에서 동시에 정보 수집"""
        print(f"📊 {len(urls)}개 사이트에서 기술적 분석 정보 수집 중...")
        
        # 비동기 수집 실행
        async def run_collection():
            tasks = [self.collect_single_site(url) for url in urls[:max_concurrent]]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        try:
            asyncio.run(run_collection())
        except Exception as e:
            print(f"⚠️ 비동기 수집 중 오류: {e}")
    
    def generate_report(self):
        """수집된 데이터 리포트 생성"""
        if not self.collected_data:
            print("📋 수집된 데이터가 없습니다.")
            return
        
        print(f"\n📈 기술적 분석 정보 수집 리포트 (총 {len(self.collected_data)}개 사이트)")
        print("=" * 60)
        
        # 키워드 통계
        all_keywords = {'indicators': [], 'patterns': [], 'concepts': []}
        
        for data in self.collected_data:
            keywords = data['keywords']
            for category in all_keywords:
                all_keywords[category].extend(keywords.get(category, []))
        
        # 가장 많이 언급된 키워드들
        for category, words in all_keywords.items():
            if words:
                try:
                    word_count = pd.Series(words).value_counts()
                    print(f"\n📊 {category.upper()} 키워드 TOP 5:")
                    for word, count in word_count.head().items():
                        print(f"  • {word}: {count}회")
                except:
                    print(f"\n📊 {category.upper()} 키워드: {', '.join(set(words))}")
        
        # 개별 사이트 정보
        print(f"\n🔍 수집된 사이트별 상세 정보:")
        for i, data in enumerate(self.collected_data, 1):
            print(f"\n{i}. {data['title'][:50]}...")
            print(f"   URL: {data['source']}")
            print(f"   내용 길이: {data['content_length']:,}자")
            print(f"   수집 시간: {data['timestamp']}")
            
            # 발견된 키워드 요약
            keywords = data['keywords']
            total_keywords = sum(len(words) for words in keywords.values())
            if total_keywords > 0:
                print(f"   기술적 분석 키워드: {total_keywords}개 발견")

# 기술적 분석 수집기 인스턴스 생성 및 실행
collector = TechnicalAnalysisCollector()

# 테스트용 URL들 (실제 기술적 분석 사이트들)
test_urls = [
    "https://www.example.com/",
    "https://httpbin.org/html",
    "https://httpbin.org/json"
]

collector.collect_from_sites(test_urls)
collector.generate_report()
print()

# === 3. 한국 주식 기술적 분석 블로그 콘텐츠 추출 ===
print("=== 3. 한국 주식 기술적 분석 블로그 콘텐츠 추출 ===")

def extract_korean_technical_content(url, title=""):
    """한국 기술적 분석 블로그에서 핵심 내용 추출"""
    try:
        loader = WebBaseLoader(url)
        loader.requests_kwargs = {
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
                'Accept-Charset': 'utf-8'
            },
            'timeout': 15
        }
        
        docs = loader.load()
        
        if docs:
            content = docs[0].page_content
            
            # 한국어 기술적 분석 관련 용어 추출
            korean_terms = {
                'RSI': ['상대강도지수', 'RSI', '과매수', '과매도', '70선', '30선'],
                'MACD': ['MACD', '이동평균수렴확산', '시그널라인', '히스토그램'],
                'Bollinger': ['볼린저밴드', '표준편차', '상한선', '하한선', '스퀴즈'],
                'Stochastic': ['스토캐스틱', '%K', '%D', '오실레이터'],
                'Moving Average': ['이동평균', '단순이동평균', '지수이동평균', '정배열', '역배열'],
                'Support_Resistance': ['지지선', '저항선', '추세선', '돌파', '이탈'],
                'Patterns': ['삼각형패턴', '플래그패턴', '쐐기형', '헤드앤숄더', '더블탑', '더블바텀']
            }
            
            found_terms = {}
            for category, terms in korean_terms.items():
                found_terms[category] = []
                for term in terms:
                    if term in content:
                        found_terms[category].append(term)
            
            # 내용 요약
            summary = {
                'title': title,
                'url': url,
                'content_length': len(content),
                'found_terms': found_terms,
                'key_sentences': []
            }
            
            # 중요한 문장 추출 (RSI, MACD 등이 포함된 문장)
            sentences = content.split('.')
            for sentence in sentences[:20]:  # 처음 20개 문장만 확인
                sentence = sentence.strip()
                if any(term in sentence for term_list in korean_terms.values() for term in term_list):
                    if len(sentence) > 10 and len(sentence) < 200:
                        summary['key_sentences'].append(sentence)
            
            return summary
            
    except Exception as e:
        print(f"❌ {title} 추출 실패: {str(e)}")
        return None

# 한국 기술적 분석 콘텐츠 예제 (공개적으로 접근 가능한 사이트들)
korean_analysis_examples = [
    ("https://httpbin.org/html", "HTML 예제"),
    ("https://www.example.com/", "기본 예제"),
]

print("📚 한국 주식 기술적 분석 콘텐츠 추출 결과:")
for url, title in korean_analysis_examples:
    print(f"\n🔍 {title} 분석 중...")
    result = extract_korean_technical_content(url, title)
    
    if result:
        print(f"✅ 성공: {result['content_length']:,}자 추출")
        print(f"📊 발견된 기술적 분석 용어:")
        
        for category, terms in result['found_terms'].items():
            if terms:
                print(f"  • {category}: {', '.join(terms)}")
        
        if result['key_sentences']:
            print(f"🎯 핵심 문장 (상위 3개):")
            for i, sentence in enumerate(result['key_sentences'][:3], 1):
                print(f"  {i}. {sentence[:100]}...")
    else:
        print("❌ 추출 실패")

print()

# === 4. 실시간 기술적 분석 데이터 모니터링 (수정됨) ===
print("=== 4. 실시간 기술적 분석 데이터 모니터링 ===")

class TechnicalAnalysisMonitor:
    """기술적 분석 데이터를 실시간으로 모니터링하는 클래스"""
    
    def __init__(self):
        self.monitoring_sites = [
            "https://httpbin.org/json",
            "https://httpbin.org/html",
            "https://www.example.com/"
        ]
        self.collected_indicators = {}
    
    async def fetch_market_data(self, url):
        """개별 사이트에서 시장 데이터 수집 (수정된 버전)"""
        try:
            loader = WebBaseLoader(url)
            loader.requests_kwargs = {
                'headers': {
                    'User-Agent': 'Technical-Analysis-Bot/1.0',
                    'Accept': 'text/html,application/json'
                },
                'timeout': 10
            }
            
            # 동기 방식으로 문서 로드
            docs = loader.load()
            
            # 가상의 기술적 지표 데이터 시뮬레이션
            import random
            
            indicators = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source': url,
                'RSI': round(random.uniform(20, 80), 2),
                'MACD': round(random.uniform(-2, 2), 4),
                'Bollinger_Upper': round(random.uniform(50000, 60000), 0),
                'Bollinger_Lower': round(random.uniform(40000, 50000), 0),
                'Volume': random.randint(1000000, 5000000),
                'status': 'success' if docs else 'failed'
            }
            
            return indicators
            
        except Exception as e:
            return {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source': url,
                'status': 'error',
                'error': str(e)
            }
    
    async def monitor_multiple_sources(self):
        """여러 소스를 동시에 모니터링"""
        print("📊 실시간 기술적 분석 데이터 모니터링 시작...")
        
        tasks = [self.fetch_market_data(url) for url in self.monitoring_sites]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 데이터 스냅샷:")
        print("-" * 80)
        
        for i, result in enumerate(results):
            if isinstance(result, dict) and result.get('status') == 'success':
                print(f"📈 소스 {i+1}: {result['source']}")
                print(f"   RSI: {result['RSI']} | MACD: {result['MACD']} | 거래량: {result['Volume']:,}")
                print(f"   볼린저밴드: {result['Bollinger_Lower']:,} ~ {result['Bollinger_Upper']:,}")
                
                # RSI 기반 매매 신호
                if result['RSI'] > 70:
                    print("   🔴 과매수 구간 - 매도 고려")
                elif result['RSI'] < 30:
                    print("   🟢 과매도 구간 - 매수 고려")
                else:
                    print("   🟡 중립 구간")
                
            else:
                print(f"❌ 소스 {i+1}: 데이터 수집 실패")
                if isinstance(result, dict) and 'error' in result:
                    print(f"   오류: {result['error']}")
            
            print()

# 실시간 모니터링 실행
monitor = TechnicalAnalysisMonitor()
asyncio.run(monitor.monitor_multiple_sources())
print()

# === 5. 기술적 분석 교육 콘텐츠 추출 및 정리 ===
print("=== 5. 기술적 분석 교육 콘텐츠 추출 및 정리 ===")

class TechnicalAnalysisEducator:
    """기술적 분석 교육 콘텐츠를 수집하고 정리하는 클래스"""
    
    def __init__(self):
        self.educational_content = {}
        self.learning_modules = {
            'RSI': {
                'description': '상대강도지수 - 과매수/과매도 상태 판단',
                'key_levels': [30, 50, 70],
                'interpretation': {
                    'above_70': '과매수 - 매도 고려',
                    'below_30': '과매도 - 매수 고려',
                    'around_50': '중립 - 추세 확인 필요'
                }
            },
            'MACD': {
                'description': '이동평균수렴확산 - 추세 전환점 포착',
                'components': ['MACD선', '시그널선', '히스토그램'],
                'signals': {
                    'golden_cross': 'MACD선이 시그널선을 상향 돌파 - 매수신호',
                    'dead_cross': 'MACD선이 시그널선을 하향 돌파 - 매도신호'
                }
            },
            'BollingerBands': {
                'description': '볼린저밴드 - 변동성과 지지/저항 수준',
                'components': ['중심선(20일 이평)', '상한선(+2σ)', '하한선(-2σ)'],
                'strategies': {
                    'band_walk': '상한선 근처 유지시 강한 상승추세',
                    'squeeze': '밴드폭 축소시 큰 움직임 예고',
                    'reversal': '밴드 터치 후 반대방향 이동'
                }
            }
        }
    
    def create_educational_summary(self):
        """교육용 기술적 분석 요약 생성"""
        print("📚 기술적 분석 학습 가이드")
        print("=" * 50)
        
        for indicator, info in self.learning_modules.items():
            print(f"\n📊 {indicator}")
            print(f"   정의: {info['description']}")
            
            if 'key_levels' in info:
                print(f"   주요 수준: {', '.join(map(str, info['key_levels']))}")
            
            if 'interpretation' in info:
                print("   해석:")
                for condition, meaning in info['interpretation'].items():
                    print(f"     • {condition.replace('_', ' ').title()}: {meaning}")
            
            if 'signals' in info:
                print("   주요 신호:")
                for signal, meaning in info['signals'].items():
                    print(f"     • {signal.replace('_', ' ').title()}: {meaning}")
            
            if 'strategies' in info:
                print("   전략:")
                for strategy, description in info['strategies'].items():
                    print(f"     • {strategy.replace('_', ' ').title()}: {description}")
    
    def extract_learning_content(self, url):
        """학습 콘텐츠에서 핵심 정보 추출"""
        try:
            loader = WebBaseLoader(url)
            loader.requests_kwargs = {
                'headers': {'User-Agent': 'Educational-Content-Extractor/1.0'},
                'timeout': 10
            }
            
            docs = loader.load()
            
            if docs:
                content = docs[0].page_content
                
                # 교육적 키워드 추출
                educational_keywords = [
                    '설명', '방법', '계산', '공식', '예제', '활용', '전략', 
                    '주의사항', '장점', '단점', '한계', '보완'
                ]
                
                educational_sentences = []
                sentences = content.split('.')
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if any(keyword in sentence for keyword in educational_keywords):
                        if 20 < len(sentence) < 150:
                            educational_sentences.append(sentence)
                
                return {
                    'url': url,
                    'educational_content': educational_sentences[:5],  # 상위 5개
                    'content_quality': len(educational_sentences)
                }
        
        except Exception as e:
            print(f"❌ 학습 콘텐츠 추출 실패: {str(e)}")
            return None
    
    def generate_practice_scenarios(self):
        """실습 시나리오 생성"""
        print("\n🎯 실습 시나리오")
        print("-" * 30)
        
        scenarios = [
            {
                'situation': 'RSI 75, 주가 신고점 근처',
                'analysis': '과매수 상태, 단기 조정 가능성 높음',
                'action': '분할매도 또는 관망'
            },
            {
                'situation': 'MACD 골든크로스 발생, 거래량 증가',
                'analysis': '상승 추세 전환 신호',
                'action': '매수 포지션 고려'
            },
            {
                'situation': '볼린저밴드 하한선 터치, RSI 25',
                'analysis': '과매도 + 지지선 도달',
                'action': '반등 매수 기회'
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"{i}. 상황: {scenario['situation']}")
            print(f"   분석: {scenario['analysis']}")
            print(f"   대응: {scenario['action']}\n")

# 교육 콘텐츠 생성 및 실행
educator = TechnicalAnalysisEducator()
educator.create_educational_summary()

# 실제 교육 사이트에서 콘텐츠 추출 (예제)
sample_educational_sites = [
    "https://www.example.com/",
    "https://httpbin.org/html"
]

print(f"\n🔍 교육 콘텐츠 품질 분석:")
for url in sample_educational_sites:
    result = educator.extract_learning_content(url)
    if result:
        print(f"✅ {url}: 교육 품질 점수 {result['content_quality']}")
        if result['educational_content']:
            print(f"   📝 핵심 내용: {result['educational_content'][0][:80]}...")

educator.generate_practice_scenarios()
print()

# === 나머지 기능들 (기존과 동일) ===

# 6. XML 파서 사용
print("=== 6. XML 파서 사용 ===")
xml_loader = WebBaseLoader("https://httpbin.org/xml")
xml_loader.default_parser = "xml"
try:
    xml_docs = xml_loader.load()
    print(f"XML 문서 로드 완료: {len(xml_docs)}개")
    print(f"XML 내용 (처음 200자): {xml_docs[0].page_content[:200]}")
except Exception as e:
    print(f"XML 로드 중 오류: {e}")
print()

# 7. Lazy Loading (메모리 효율적)
print("=== 7. Lazy Loading ===")
loader_lazy = WebBaseLoader("https://www.example.com/")

pages = []
for doc in loader_lazy.lazy_load():
    pages.append(doc)
    print(f"Lazy load로 문서 로드: {doc.metadata.get('title', 'N/A')}")
print()

# 8. 비동기 Lazy Loading (수정됨)
async def async_lazy_load_example():
    print("=== 8. 비동기 Lazy Loading ===")
    loader_async_lazy = WebBaseLoader("https://www.example.com/")
    
    pages_async = []
    async for doc in loader_async_lazy.alazy_load():
        pages_async.append(doc)
        print(f"Async lazy load로 문서 로드: {doc.metadata.get('title', 'N/A')}")
    
    return pages_async

# 비동기 lazy loading 실행
try:
    pages_async = asyncio.run(async_lazy_load_example())
    print(f"비동기 lazy loading 완료: {len(pages_async)}개 문서")
except Exception as e:
    print(f"비동기 lazy loading 중 오류: {e}")
print()

# 9. 에러 처리 예제
print("=== 9. 에러 처리 예제 ===")
def safe_load_web_content(url):
    """안전한 웹 컨텐츠 로딩 함수"""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs
    except Exception as e:
        print(f"URL {url} 로딩 중 오류 발생: {e}")
        return None

# 유효하지 않은 URL 테스트
invalid_docs = safe_load_web_content("https://nonexistent-site-12345.com")
valid_docs = safe_load_web_content("https://www.example.com/")

if valid_docs:
    print("유효한 URL로 문서 로드 성공")
print()

# 10. 문서 메타데이터 분석
print("=== 10. 문서 메타데이터 분석 ===")
loader_meta = WebBaseLoader("https://www.example.com/")
docs_meta = loader_meta.load()

if docs_meta:
    doc = docs_meta[0]
    print("문서 메타데이터:")
    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")
    
    print(f"\n문서 내용 길이: {len(doc.page_content)} 문자")
    print(f"문서 내용 (처음 100자): {doc.page_content[:100]}...")
print()

# === 추가 기능: 고급 에러 처리 및 재시도 로직 ===
print("=== 11. 고급 에러 처리 및 재시도 로직 ===")

import time
from typing import Optional, List

class RobustWebLoader:
    """견고한 웹 로더 클래스 - 재시도 및 에러 처리 포함"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def load_with_retry(self, url: str, description: str = "") -> Optional[List]:
        """재시도 로직이 포함된 문서 로드"""
        for attempt in range(self.max_retries):
            try:
                print(f"🔄 시도 {attempt + 1}/{self.max_retries}: {description or url}")
                
                loader = WebBaseLoader(url)
                loader.requests_kwargs = {
                    'headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    },
                    'timeout': 10 + (attempt * 5),  # 재시도마다 타임아웃 증가
                    'verify': True
                }
                
                docs = loader.load()
                
                if docs and len(docs) > 0:
                    print(f"✅ 성공: {len(docs)}개 문서, {len(docs[0].page_content)}자")
                    return docs
                else:
                    print(f"⚠️ 빈 문서 반환")
                    
            except Exception as e:
                print(f"❌ 시도 {attempt + 1} 실패: {str(e)[:100]}...")
                
                if attempt < self.max_retries - 1:
                    print(f"⏳ {self.retry_delay}초 후 재시도...")
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 1.5  # 지수 백오프
                else:
                    print(f"💥 모든 재시도 실패: {url}")
        
        return None
    
    async def async_load_with_retry(self, url: str, description: str = "") -> Optional[List]:
        """비동기 재시도 로직"""
        for attempt in range(self.max_retries):
            try:
                print(f"🔄 비동기 시도 {attempt + 1}/{self.max_retries}: {description or url}")
                
                loader = WebBaseLoader(url)
                loader.requests_kwargs = {
                    'headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    },
                    'timeout': 10 + (attempt * 5),
                }
                
                # 올바른 비동기 처리
                docs = []
                async for doc in loader.alazy_load():
                    docs.append(doc)
                
                if docs and len(docs) > 0:
                    print(f"✅ 비동기 성공: {len(docs)}개 문서")
                    return docs
                    
            except Exception as e:
                print(f"❌ 비동기 시도 {attempt + 1} 실패: {str(e)[:100]}...")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    self.retry_delay *= 1.5
        
        return None

# 견고한 로더 테스트
robust_loader = RobustWebLoader(max_retries=2, retry_delay=1.0)

test_urls_with_descriptions = [
    ("https://www.example.com/", "예제 사이트"),
    ("https://httpbin.org/html", "HTTP 테스트"),
    ("https://nonexistent-really-fake-url.com/", "존재하지 않는 사이트")
]

for url, desc in test_urls_with_descriptions:
    result = robust_loader.load_with_retry(url, desc)
    if result:
        print(f"📊 {desc}: 최종 성공\n")
    else:
        print(f"💔 {desc}: 최종 실패\n")

# === 12. 배치 처리 및 병렬 실행 ===
print("=== 12. 배치 처리 및 병렬 실행 ===")

class BatchWebLoader:
    """배치 처리 웹 로더"""
    
    def __init__(self, max_concurrent: int = 3, delay_between_batches: float = 1.0):
        self.max_concurrent = max_concurrent
        self.delay_between_batches = delay_between_batches
    
    async def process_urls_in_batches(self, urls: List[str], batch_size: int = 3):
        """URL들을 배치로 나누어 처리"""
        print(f"📦 총 {len(urls)}개 URL을 {batch_size}개씩 배치 처리")
        
        results = []
        
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(urls) + batch_size - 1) // batch_size
            
            print(f"\n🔄 배치 {batch_num}/{total_batches} 처리 중...")
            
            # 배치 내 URL들을 병렬 처리
            batch_tasks = [self.load_single_url(url) for url in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 결과 수집
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"❌ 배치 {batch_num} URL {j+1}: {str(result)[:50]}...")
                    results.append(None)
                else:
                    results.append(result)
            
            # 배치 간 지연
            if i + batch_size < len(urls):
                print(f"⏳ 다음 배치까지 {self.delay_between_batches}초 대기...")
                await asyncio.sleep(self.delay_between_batches)
        
        # 결과 요약
        successful = len([r for r in results if r is not None])
        print(f"\n📈 배치 처리 완료: {successful}/{len(urls)} 성공")
        
        return results
    
    async def load_single_url(self, url: str):
        """단일 URL 로드"""
        try:
            loader = WebBaseLoader(url)
            loader.requests_kwargs = {
                'headers': {
                    'User-Agent': 'Batch-Web-Loader/1.0'
                },
                'timeout': 10
            }
            
            docs = []
            async for doc in loader.alazy_load():
                docs.append(doc)
            
            if docs:
                print(f"✅ {url}: {len(docs[0].page_content)}자")
                return {
                    'url': url,
                    'docs': docs,
                    'status': 'success'
                }
            else:
                print(f"⚠️ {url}: 빈 응답")
                return None
                
        except Exception as e:
            print(f"❌ {url}: {str(e)[:50]}...")
            raise e

# 배치 로더 테스트
batch_loader = BatchWebLoader(max_concurrent=2, delay_between_batches=0.5)

batch_test_urls = [
    "https://www.example.com/",
    "https://httpbin.org/html", 
    "https://httpbin.org/json",
    "https://httpbin.org/xml",
    "https://httpbin.org/user-agent"
]

batch_results = asyncio.run(batch_loader.process_urls_in_batches(batch_test_urls, batch_size=2))

# === 13. 성능 모니터링 및 통계 ===
print("\n=== 13. 성능 모니터링 및 통계 ===")

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_bytes': 0,
            'total_time': 0,
            'avg_response_time': 0,
            'fastest_request': float('inf'),
            'slowest_request': 0
        }
    
    def record_request(self, success: bool, response_time: float, content_size: int = 0):
        """요청 기록"""
        self.stats['total_requests'] += 1
        self.stats['total_time'] += response_time
        
        if success:
            self.stats['successful_requests'] += 1
            self.stats['total_bytes'] += content_size
            self.stats['fastest_request'] = min(self.stats['fastest_request'], response_time)
            self.stats['slowest_request'] = max(self.stats['slowest_request'], response_time)
        else:
            self.stats['failed_requests'] += 1
        
        # 평균 응답 시간 계산
        if self.stats['total_requests'] > 0:
            self.stats['avg_response_time'] = self.stats['total_time'] / self.stats['total_requests']
    
    def get_report(self) -> str:
        """성능 리포트 생성"""
        success_rate = (self.stats['successful_requests'] / max(1, self.stats['total_requests'])) * 100
        avg_content_size = self.stats['total_bytes'] / max(1, self.stats['successful_requests'])
        
        report = f"""
📊 성능 모니터링 리포트
{'='*50}
총 요청 수: {self.stats['total_requests']}
성공한 요청: {self.stats['successful_requests']}
실패한 요청: {self.stats['failed_requests']}
성공률: {success_rate:.1f}%

⏱️ 응답 시간 통계:
  평균: {self.stats['avg_response_time']:.2f}초
  최고속: {self.stats['fastest_request']:.2f}초
  최저속: {self.stats['slowest_request']:.2f}초

📦 데이터 통계:
  총 바이트: {self.stats['total_bytes']:,}
  평균 콘텐츠 크기: {avg_content_size:,.0f} 바이트
  총 처리 시간: {self.stats['total_time']:.2f}초
"""
        return report

# 성능 모니터링 테스트
monitor = PerformanceMonitor()

test_performance_urls = [
    "https://www.example.com/",
    "https://httpbin.org/html",
    "https://httpbin.org/delay/1"  # 1초 지연 URL
]

print("🔍 성능 테스트 시작...")

for url in test_performance_urls:
    start_time = time.time()
    
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        end_time = time.time()
        response_time = end_time - start_time
        content_size = len(docs[0].page_content) if docs else 0
        
        monitor.record_request(success=True, response_time=response_time, content_size=content_size)
        print(f"✅ {url}: {response_time:.2f}초, {content_size:,}바이트")
        
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        
        monitor.record_request(success=False, response_time=response_time)
        print(f"❌ {url}: {response_time:.2f}초, 실패")

print(monitor.get_report())

print("\n" + "="*80)
print("🎊 수정된 WebBaseLoader 샘플 코드 실행 완료!")
print("="*80)
print("""
🔧 주요 수정 사항:
   ✅ aload() 비동기 처리 오류 해결
   ✅ async for 루프를 사용한 올바른 비동기 처리
   ✅ 재시도 로직 및 견고한 에러 처리
   ✅ 배치 처리 및 병렬 실행 기능
   ✅ 성능 모니터링 및 통계 수집
   ✅ 경고 메시지 필터링

💡 핵심 개선점:
   • aload() → async for alazy_load() 패턴 사용
   • 예외 처리 강화 및 재시도 메커니즘
   • 배치 처리로 대량 URL 효율적 처리
   • 실시간 성능 모니터링
   • 서버 부하 방지를 위한 지연 로직

🚀 실무 활용 팁:
   • 대량 데이터 수집 시 배치 처리 활용
   • 불안정한 네트워크 환경에서 재시도 로직 필수
   • 성능 모니터링으로 병목 지점 파악
   • 서버 정책에 맞는 요청 빈도 조절
""")