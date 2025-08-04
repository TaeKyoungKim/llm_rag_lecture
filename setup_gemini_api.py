"""
Gemini API 키 설정 도우미 스크립트
Gemini 임베딩을 사용하기 위한 API 키 설정을 도와줍니다.
"""

import os
import sys
from pathlib import Path

def check_gemini_api_key():
    """Gemini API 키 확인"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if api_key:
        print("✅ GOOGLE_API_KEY 환경 변수가 설정되어 있습니다.")
        print(f"   키 길이: {len(api_key)}자")
        print(f"   키 시작: {api_key[:10]}...")
        return True
    else:
        print("❌ GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        return False

def setup_gemini_api_key():
    """Gemini API 키 설정"""
    print("🔧 Gemini API 키 설정")
    print("=" * 50)
    
    print("\n📋 Gemini API 키 설정 방법:")
    print("1. Google AI Studio (https://makersuite.google.com/app/apikey) 방문")
    print("2. Google 계정으로 로그인")
    print("3. 'Create API Key' 클릭")
    print("4. 생성된 API 키를 복사")
    print("5. 아래에 입력하거나 환경 변수로 설정")
    
    # API 키 입력 받기
    print("\n🔑 API 키를 입력하세요 (또는 Enter로 건너뛰기):")
    api_key = input("API Key: ").strip()
    
    if api_key:
        # 환경 변수 설정
        os.environ["GOOGLE_API_KEY"] = api_key
        print("✅ API 키가 환경 변수로 설정되었습니다.")
        
        # .env 파일 생성 (선택사항)
        create_env_file = input("\n📁 .env 파일을 생성하시겠습니까? (y/n): ").strip().lower()
        if create_env_file == 'y':
            create_env_file_with_key(api_key)
        
        return True
    else:
        print("⚠️ API 키를 입력하지 않았습니다. HuggingFace 임베딩을 사용합니다.")
        return False

def create_env_file_with_key(api_key):
    """환경 변수 파일 생성"""
    env_file = Path(".env")
    
    try:
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(f"GOOGLE_API_KEY={api_key}\n")
        print(f"✅ .env 파일이 생성되었습니다: {env_file}")
        print("💡 이제 python-dotenv를 사용하여 자동으로 로드할 수 있습니다.")
    except Exception as e:
        print(f"❌ .env 파일 생성 실패: {str(e)}")

def test_gemini_connection():
    """Gemini API 연결 테스트"""
    print("\n🧪 Gemini API 연결 테스트")
    print("-" * 30)
    
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ API 키가 설정되지 않았습니다.")
            return False
        
        print("🔗 Gemini API에 연결 중...")
        
        # 임베딩 모델 초기화
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # 간단한 테스트
        test_text = "Hello, this is a test for Gemini embeddings."
        embedding = embeddings.embed_query(test_text)
        
        if embedding and len(embedding) > 0:
            print(f"✅ 연결 성공! 임베딩 차원: {len(embedding)}")
            print(f"   테스트 임베딩 샘플: {embedding[:5]}...")
            return True
        else:
            print("❌ 임베딩 생성 실패")
            return False
            
    except Exception as e:
        print(f"❌ 연결 테스트 실패: {str(e)}")
        
        # 오류 유형별 안내
        error_msg = str(e).lower()
        if "invalid_grant" in error_msg or "bad request" in error_msg:
            print("💡 API 키 인증 오류입니다. API 키를 확인해주세요.")
        elif "timeout" in error_msg:
            print("💡 네트워크 타임아웃입니다. 인터넷 연결을 확인해주세요.")
        elif "quota" in error_msg:
            print("💡 API 할당량 초과입니다. 잠시 후 다시 시도하세요.")
        
        return False

def show_usage_examples():
    """사용 예시 보여주기"""
    print("\n📖 사용 예시")
    print("=" * 30)
    
    print("\n1️⃣ 환경 변수로 설정:")
    print("   Windows (CMD):")
    print("   set GOOGLE_API_KEY=your_api_key_here")
    print("   ")
    print("   Windows (PowerShell):")
    print("   $env:GOOGLE_API_KEY='your_api_key_here'")
    print("   ")
    print("   Linux/Mac:")
    print("   export GOOGLE_API_KEY=your_api_key_here")
    
    print("\n2️⃣ .env 파일 사용:")
    print("   # .env 파일 생성")
    print("   GOOGLE_API_KEY=your_api_key_here")
    print("   ")
    print("   # Python에서 로드")
    print("   from dotenv import load_dotenv")
    print("   load_dotenv()")
    
    print("\n3️⃣ 교육용 시스템 실행:")
    print("   # HuggingFace 임베딩 (기본)")
    print("   python educational_faiss_system.py")
    print("   ")
    print("   # Gemini 임베딩")
    print("   python educational_faiss_system.py gemini")

def main():
    """메인 함수"""
    print("🎓 Gemini API 키 설정 도우미")
    print("=" * 50)
    
    # 현재 API 키 상태 확인
    has_api_key = check_gemini_api_key()
    
    if not has_api_key:
        # API 키 설정
        setup_success = setup_gemini_api_key()
        if setup_success:
            has_api_key = True
    
    if has_api_key:
        # 연결 테스트
        test_success = test_gemini_connection()
        if test_success:
            print("\n🎉 Gemini API 설정이 완료되었습니다!")
            print("이제 'python educational_faiss_system.py gemini'로 실행할 수 있습니다.")
        else:
            print("\n⚠️ Gemini API 연결에 실패했습니다.")
            print("HuggingFace 임베딩을 사용하거나 API 키를 다시 확인해주세요.")
    
    # 사용 예시 보여주기
    show_usage_examples()
    
    print("\n💡 추가 도움이 필요하시면 다음을 확인해주세요:")
    print("   - Google AI Studio: https://makersuite.google.com/app/apikey")
    print("   - Gemini API 문서: https://ai.google.dev/docs")
    print("   - LangChain Gemini 문서: https://python.langchain.com/docs/integrations/platforms/google")

if __name__ == "__main__":
    main() 