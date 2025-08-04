import base64
import os
from datetime import datetime
from PIL import Image
import io

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# Gemini 이미지 생성 모델 초기화
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-preview-image-generation")

def generate_and_save_image(prompt, save_directory="generated_images"):
    """
    이미지를 생성하고 로컬에 저장하는 함수
    
    Args:
        prompt (str): 이미지 생성 프롬프트
        save_directory (str): 이미지를 저장할 디렉토리
    
    Returns:
        str: 저장된 이미지 파일 경로
    """
    
    # 저장 디렉토리 생성
    os.makedirs(save_directory, exist_ok=True)
    
    print(f"🎨 이미지 생성 중: {prompt}")
    print("⏳ 잠시만 기다려주세요...")
    
    # 메시지 구성
    message = {
        "role": "user",
        "content": prompt,
    }
    
    try:
        # Gemini로 이미지 생성
        response = llm.invoke(
            [message],
            generation_config=dict(response_modalities=["TEXT", "IMAGE"]),
        )
        
        # 응답에서 이미지 추출
        image_base64 = _get_image_base64(response)
        
        # 파일명 생성 (타임스탬프 포함)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gemini_image_{timestamp}.png"
        filepath = os.path.join(save_directory, filename)
        
        # Base64를 이미지로 디코딩하여 저장
        image_data = base64.b64decode(image_base64)
        
        # PIL을 사용하여 이미지 저장
        image = Image.open(io.BytesIO(image_data))
        image.save(filepath, "PNG")
        
        print(f"✅ 이미지가 성공적으로 저장되었습니다: {filepath}")
        
        # 이미지 정보 출력
        print(f"📋 이미지 정보:")
        print(f"   - 크기: {image.size}")
        print(f"   - 모드: {image.mode}")
        print(f"   - 파일 크기: {len(image_data)} bytes")
        
        # 이미지 뷰어로 열기 (선택사항)
        try:
            # Windows
            if os.name == 'nt':
                os.system(f'start {filepath}')
            # macOS
            elif os.name == 'posix' and os.uname().sysname == 'Darwin':
                os.system(f'open {filepath}')
            # Linux
            else:
                os.system(f'xdg-open {filepath}')
            print("🖼️  기본 이미지 뷰어로 이미지를 열었습니다.")
        except:
            print("💡 이미지 뷰어를 자동으로 열 수 없습니다. 직접 파일을 열어주세요.")
        
        return filepath
        
    except Exception as e:
        print(f"❌ 이미지 생성 중 오류가 발생했습니다: {e}")
        return None

def _get_image_base64(response: AIMessage) -> str:
    """
    AIMessage 응답에서 Base64 이미지 데이터를 추출하는 함수
    
    Args:
        response (AIMessage): Gemini 응답 메시지
    
    Returns:
        str: Base64 인코딩된 이미지 데이터
    """
    try:
        image_block = next(
            block
            for block in response.content
            if isinstance(block, dict) and block.get("image_url")
        )
        return image_block["image_url"].get("url").split(",")[-1]
    except StopIteration:
        raise ValueError("응답에서 이미지를 찾을 수 없습니다.")
    except Exception as e:
        raise ValueError(f"이미지 추출 중 오류 발생: {e}")

def batch_generate_images(prompts, save_directory="generated_images"):
    """
    여러 프롬프트로 배치 이미지 생성
    
    Args:
        prompts (list): 이미지 생성 프롬프트 리스트
        save_directory (str): 이미지를 저장할 디렉토리
    
    Returns:
        list: 생성된 이미지 파일 경로들
    """
    generated_files = []
    
    print(f"🎨 {len(prompts)}개의 이미지를 배치 생성합니다...")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n📝 {i}/{len(prompts)}: {prompt}")
        filepath = generate_and_save_image(prompt, save_directory)
        if filepath:
            generated_files.append(filepath)
    
    print(f"\n🎉 배치 생성 완료! 총 {len(generated_files)}개의 이미지가 생성되었습니다.")
    return generated_files

# 메인 실행 부분
if __name__ == "__main__":
    print("🚀 Gemini 이미지 생성기")
    print("=" * 50)
    
    # 예제 프롬프트들
    example_prompts = [
        "Generate a photorealistic image of a cuddly cat wearing a hat.",
        "Create a beautiful sunset landscape with mountains and a lake.",
        "Draw a futuristic city with flying cars and neon lights.",
        "Make an image of a cozy coffee shop with warm lighting.",
        "Generate a magical forest with glowing mushrooms and fairy lights."
    ]
    
    print("📋 사용 가능한 예제 프롬프트:")
    for i, prompt in enumerate(example_prompts, 1):
        print(f"{i}. {prompt}")
    
    print("\n선택하세요:")
    print("1-5: 예제 프롬프트 선택")
    print("6: 모든 예제 프롬프트로 배치 생성")
    print("7: 직접 프롬프트 입력")
    print("0: 종료")
    
    while True:
        try:
            choice = input("\n선택 (0-7): ").strip()
            
            if choice == "0":
                print("👋 프로그램을 종료합니다.")
                break
            
            elif choice in ["1", "2", "3", "4", "5"]:
                selected_prompt = example_prompts[int(choice) - 1]
                generate_and_save_image(selected_prompt)
            
            elif choice == "6":
                batch_generate_images(example_prompts)
            
            elif choice == "7":
                custom_prompt = input("이미지 생성 프롬프트를 입력하세요: ").strip()
                if custom_prompt:
                    generate_and_save_image(custom_prompt)
                else:
                    print("프롬프트를 입력해주세요.")
            
            else:
                print("올바른 번호를 선택해주세요 (0-7).")
                
        except KeyboardInterrupt:
            print("\n\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류가 발생했습니다: {e}")