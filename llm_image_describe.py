import base64
from pathlib import Path
import mimetypes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# 환경변수 로드 및 모델 초기화
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")

# 한글 출력 시스템 프롬프트 + LCEL 체인
image_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 이미지 분석 전문가입니다. 모든 응답은 반드시 한글로 작성해주세요."),
    ("human", [
        {"type": "text", "text": "{text}"},
        {"type": "image_url", "image_url": "{image_url}"}
    ])
])

# LCEL 체인 구성
image_chain = image_prompt | llm | StrOutputParser()

def analyze_url_image(url, text="이미지를 한글로 설명해주세요."):
    """URL 이미지 분석"""
    return image_chain.invoke({"text": text, "image_url": url})

def analyze_local_image(image_path, text="이미지를 한글로 설명해주세요."):
    """로컬 이미지 분석"""
    # 파일을 Base64로 인코딩
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    
    # MIME 타입 설정
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream" # Fallback
    # 체인 실행
    return image_chain.invoke({
        "text": text,
        "image_url": f"data:{mime_type};base64,{encoded}"
    })

# 실행 예제
if __name__ == "__main__":
    print("🔗 LCEL 이미지 분석 (한글 출력)")
    print("=" * 40)
    
    # URL 이미지 분석
    print("🌐 URL 이미지:")
    url_result = analyze_url_image("https://picsum.photos/400/300")
    print(f"결과: {url_result}\n")
    
    # 로컬 이미지 분석 (있는 경우)
    image_files = list(Path(".").glob("*.png")) + list(Path(".").glob("*.jpg"))
    if image_files:
        print(f"📁 로컬 이미지 ({image_files[0]}):")
        local_result = analyze_local_image(str(image_files[0]))
        print(f"결과: {local_result}")
    
    # 사용자 입력
    print("\n" + "="*40)
    user_input = input("이미지 경로나 URL 입력 (Enter로 건너뛰기): ").strip()
    if user_input:
        try:
            if user_input.startswith(('http://', 'https://')):
                result = analyze_url_image(user_input)
            else:
                result = analyze_local_image(user_input)
            print(f"결과: {result}")
        except Exception as e:
            print(f"오류: {e}")