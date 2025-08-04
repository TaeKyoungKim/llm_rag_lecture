# LangChain WebBaseLoader 샘플 코드

# 필요한 패키지 설치
# pip install -qU langchain_community beautifulsoup4 nest_asyncio

import asyncio
import nest_asyncio
from langchain_community.document_loaders import WebBaseLoader

# Jupyter 환경에서 asyncio 사용 시 필요
nest_asyncio.apply()

# 1. 기본 사용법 - 단일 웹페이지 로드
print("=== 1. 기본 사용법 ===")
loader = WebBaseLoader("https://www.iana.org/domains/reserved")
docs = loader.load()

print(f"문서 개수: {len(docs)}")
print(f"제목: {docs[0].metadata.get('title', 'N/A')}")
print(f"언어: {docs[0].metadata.get('language', 'N/A')}")
print(f"내용 (처음 200자): {docs[0].page_content[:200]}")
print()

# 2. SSL 검증 우회 설정
print("=== 2. SSL 검증 우회 설정 ===")
loader_ssl = WebBaseLoader("https://www.iana.org/domains/reserved")
loader_ssl.requests_kwargs = {'verify': False}
docs_ssl = loader_ssl.load()
print("SSL 검증 우회로 문서 로드 완료")
print()

# 3. 여러 웹페이지 동시 로드
print("=== 3. 여러 웹페이지 동시 로드 ===")
urls = [
    "https://www.example.com/",
    "https://httpbin.org/html"
]

loader_multiple = WebBaseLoader(urls)
loader_multiple.requests_per_second = 2  # 초당 최대 2개 요청

docs_multiple = loader_multiple.load()
print(f"로드된 문서 개수: {len(docs_multiple)}")
for i, doc in enumerate(docs_multiple):
    print(f"문서 {i+1}: {doc.metadata.get('source', 'N/A')}")
print()


import asyncio
import nest_asyncio
from langchain_community.document_loaders import WebBaseLoader
# 5. XML 파서 사용
print("=== 5. XML 파서 사용 ===")
# XML 파일 로드 예제 (실제 XML URL 사용)
xml_loader = WebBaseLoader("https://httpbin.org/xml")
xml_loader.default_parser = "xml"
try:
    xml_docs = xml_loader.load()
    print(f"XML 문서 로드 완료: {len(xml_docs)}개")
    print(f"XML 내용 (처음 200자): {xml_docs[0].page_content[:200]}")
except Exception as e:
    print(f"XML 로드 중 오류: {e}")
print()

# 6. Lazy Loading (메모리 효율적)
print("=== 6. Lazy Loading ===")
loader_lazy = WebBaseLoader("https://www.example.com/")

pages = []
for doc in loader_lazy.lazy_load():
    pages.append(doc)
    print(f"Lazy load로 문서 로드: {doc.metadata.get('title', 'N/A')}")
print()

# 7. 비동기 Lazy Loading
async def async_lazy_load_example():
    print("=== 7. 비동기 Lazy Loading ===")
    loader_async_lazy = WebBaseLoader("https://www.example.com/")

    pages_async = []
    async for doc in loader_async_lazy.alazy_load():
        pages_async.append(doc)
        print(f"Async lazy load로 문서 로드: {doc.metadata.get('title', 'N/A')}")

    return pages_async

# 비동기 lazy loading 실행

pages_async = asyncio.run(async_lazy_load_example())
print()

# 8. 프록시 사용 (예제)
print("=== 8. 프록시 사용 예제 ===")
# 실제 프록시 정보가 있을 때 사용
"""
loader_proxy = WebBaseLoader(
    "https://www.example.com/",
    proxies={
        "http": "http://username:password@proxy.server.com:8080/",
        "https": "https://username:password@proxy.server.com:8080/",
    },
)
docs_proxy = loader_proxy.load()
"""
print("프록시 설정 예제 코드 (주석 처리됨)")
print()

# 9. 커스텀 헤더 설정
print("=== 9. 커스텀 헤더 설정 ===")
loader_headers = WebBaseLoader("https://httpbin.org/headers")
loader_headers.requests_kwargs = {
    'headers': {
        'User-Agent': 'My Custom Bot 1.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }
}

try:
    docs_headers = loader_headers.load()
    print("커스텀 헤더로 문서 로드 완료")
    print(f"응답 내용 (처음 300자): {docs_headers[0].page_content[:300]}")
except Exception as e:
    print(f"헤더 설정 중 오류: {e}")
print()

# 10. 에러 처리 예제
print("=== 10. 에러 처리 예제 ===")
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

# 11. 문서 메타데이터 분석
print("=== 11. 문서 메타데이터 분석 ===")
loader_meta = WebBaseLoader("https://www.example.com/")
docs_meta = loader_meta.load()

if docs_meta:
    doc = docs_meta[0]
    print("문서 메타데이터:")
    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")

    print(f"\n문서 내용 길이: {len(doc.page_content)} 문자")
    print(f"문서 내용 (처음 100자): {doc.page_content[:100]}...")

print("\n=== WebBaseLoader 샘플 코드 실행 완료 ===")