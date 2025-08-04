"""
교육용 FAISS 시스템 문제 진단 및 해결 스크립트
일반적인 문제들을 자동으로 진단하고 해결 방법을 제시합니다.
"""

import os
import sys
from pathlib import Path

def check_pdf_file():
    """PDF 파일 존재 확인"""
    print("📄 PDF 파일 확인 중...")
    
    pdf_path = Path("DocumentsLoader/data/기술적차트분석이론및방법.pdf")
    
    if pdf_path.exists():
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ PDF 파일 발견: {pdf_path}")
        print(f"   📊 파일 크기: {size_mb:.2f} MB")
        return True
    else:
        print(f"   ❌ PDF 파일을 찾을 수 없습니다: {pdf_path}")
        print("   💡 해결 방법:")
        print("      1. DocumentsLoader/data/ 폴더에 PDF 파일을 넣어주세요")
        print("      2. 파일명이 '기술적차트분석이론및방법.pdf'인지 확인하세요")
        return False

def check_dependencies():
    """필요한 라이브러리 확인"""
    print("\n📦 라이브러리 확인 중...")
    
    required_packages = [
        "faiss",
        "langchain",
        "langchain_community", 
        "langchain_google_genai",
        "sentence_transformers",
        "PyPDF2",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (설치 필요)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   💡 설치 명령어:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    else:
        print("   ✅ 모든 필요한 라이브러리가 설치되어 있습니다.")
        return True

def check_gemini_api():
    """Gemini API 설정 확인"""
    print("\n🔑 Gemini API 설정 확인 중...")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("   ❌ GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("   💡 해결 방법:")
        print("      1. python setup_gemini_api.py 실행")
        print("      2. 또는 수동으로 환경 변수 설정")
        return False
    
    print(f"   ✅ GOOGLE_API_KEY 설정됨 (길이: {len(api_key)}자)")
    
    # API 키 형식 확인
    if api_key.startswith("AIza"):
        print("   ✅ API 키 형식이 올바릅니다.")
    else:
        print("   ⚠️ API 키 형식이 예상과 다릅니다.")
    
    return True

def check_network_connection():
    """네트워크 연결 확인"""
    print("\n🌐 네트워크 연결 확인 중...")
    
    try:
        import urllib.request
        urllib.request.urlopen("https://www.google.com", timeout=5)
        print("   ✅ 인터넷 연결 정상")
        return True
    except Exception as e:
        print(f"   ❌ 인터넷 연결 실패: {str(e)}")
        print("   💡 네트워크 연결을 확인해주세요.")
        return False

def check_disk_space():
    """디스크 공간 확인"""
    print("\n💾 디스크 공간 확인 중...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        print(f"   📊 사용 가능한 공간: {free_gb:.2f} GB")
        
        if free_gb > 1.0:
            print("   ✅ 충분한 디스크 공간이 있습니다.")
            return True
        else:
            print("   ⚠️ 디스크 공간이 부족할 수 있습니다.")
            print("   💡 최소 1GB 이상의 여유 공간을 확보해주세요.")
            return False
    except Exception as e:
        print(f"   ⚠️ 디스크 공간 확인 실패: {str(e)}")
        return True  # 확인 실패 시 계속 진행

def test_embeddings():
    """임베딩 모델 테스트"""
    print("\n🧪 임베딩 모델 테스트 중...")
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        print("   🔄 HuggingFace 임베딩 모델 로드 중...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        test_text = "This is a test for embeddings."
        embedding = embeddings.embed_query(test_text)
        
        if embedding and len(embedding) > 0:
            print(f"   ✅ HuggingFace 임베딩 테스트 성공 (차원: {len(embedding)})")
            return True
        else:
            print("   ❌ 임베딩 생성 실패")
            return False
            
    except Exception as e:
        print(f"   ❌ 임베딩 테스트 실패: {str(e)}")
        return False

def run_full_diagnostic():
    """전체 진단 실행"""
    print("🔍 교육용 FAISS 시스템 전체 진단")
    print("=" * 50)
    
    results = []
    
    # 각 항목 확인
    results.append(("PDF 파일", check_pdf_file()))
    results.append(("라이브러리", check_dependencies()))
    results.append(("Gemini API", check_gemini_api()))
    results.append(("네트워크", check_network_connection()))
    results.append(("디스크 공간", check_disk_space()))
    results.append(("임베딩 모델", test_embeddings()))
    
    # 결과 요약
    print("\n📋 진단 결과 요약")
    print("-" * 30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for item, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"   {item}: {status}")
    
    print(f"\n📊 전체 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("\n🎉 모든 검사가 통과했습니다!")
        print("시스템을 정상적으로 실행할 수 있습니다.")
        return True
    else:
        print(f"\n⚠️ {total - passed}개 항목에 문제가 있습니다.")
        print("위의 해결 방법을 참고하여 문제를 해결해주세요.")
        return False

def suggest_solutions():
    """문제별 해결 방법 제시"""
    print("\n💡 일반적인 해결 방법")
    print("=" * 30)
    
    print("\n1️⃣ PDF 파일 문제:")
    print("   - DocumentsLoader/data/ 폴더에 PDF 파일을 넣어주세요")
    print("   - 파일명: '기술적차트분석이론및방법.pdf'")
    
    print("\n2️⃣ 라이브러리 설치:")
    print("   pip install faiss-cpu langchain langchain-community langchain-google-genai sentence-transformers PyPDF2 numpy")
    
    print("\n3️⃣ Gemini API 설정:")
    print("   python setup_gemini_api.py")
    
    print("\n4️⃣ 네트워크 문제:")
    print("   - 인터넷 연결 확인")
    print("   - 방화벽 설정 확인")
    
    print("\n5️⃣ 메모리/디스크 문제:")
    print("   - 충분한 여유 공간 확보")
    print("   - 청크 크기 줄이기")

def main():
    """메인 함수"""
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # 빠른 진단
        print("⚡ 빠른 진단 실행")
        check_pdf_file()
        check_dependencies()
        check_gemini_api()
    else:
        # 전체 진단
        success = run_full_diagnostic()
        if not success:
            suggest_solutions()

if __name__ == "__main__":
    main() 