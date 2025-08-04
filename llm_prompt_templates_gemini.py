# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# 필요한 라이브러리 임포트
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# Gemini 2.0 Flash 모델 초기화
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

print("🚀 Gemini API PromptTemplate 예제")
print("="*60)

# === 1. 기본 PromptTemplate 사용 ===
print("\n📝 1. 기본 PromptTemplate")
print("-" * 30)

# 변수를 포함한 프롬프트 템플릿 정의
prompt_template = PromptTemplate.from_template(
    "{topic}에 대해 초보자도 이해하기 쉽게 설명해줘."
)

# format() 메서드로 변수에 값을 채워 넣음
prompt = prompt_template.format(topic="벡터 데이터베이스")
print(f"생성된 프롬프트: {prompt}")

# Gemini로 응답 생성
response = llm.invoke(prompt)
print(f"응답: {response.content}")

# === 2. 다중 변수 PromptTemplate ===
print("\n📝 2. 다중 변수 PromptTemplate")
print("-" * 30)

# 여러 변수를 포함한 템플릿
multi_var_template = PromptTemplate.from_template(
    "{subject} 분야에서 {level} 수준의 학습자를 위한 {topic}에 대한 설명을 {style} 스타일로 작성해줘."
)

# 변수들에 값 할당
formatted_prompt = multi_var_template.format(
    subject="데이터 사이언스",
    level="중급자",
    topic="머신러닝 알고리즘",
    style="친근하고 유머러스한"
)

print(f"생성된 프롬프트: {formatted_prompt}")
response = llm.invoke(formatted_prompt)
print(f"응답: {response.content[:200]}...")  # 처음 200자만 출력

# === 3. ChatPromptTemplate 사용 ===
print("\n📝 3. ChatPromptTemplate (시스템 + 사용자 메시지)")
print("-" * 30)

# 시스템 메시지와 사용자 메시지를 포함한 채팅 템플릿
chat_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 {expertise} 전문가입니다. {tone} 톤으로 답변해주세요."),
    ("human", "{question}")
])

# 변수 값 설정
formatted_chat = chat_template.format_messages(
    expertise="인공지능",
    tone="친절하고 전문적인",
    question="트랜스포머 모델의 핵심 개념을 설명해주세요."
)

print(f"생성된 메시지: {formatted_chat}")
response = llm.invoke(formatted_chat)
print(f"응답: {response.content[:200]}...")

# === 4. 템플릿 체인 사용 ===
print("\n📝 4. 템플릿과 LLM 체인 결합")
print("-" * 30)

# 프롬프트 템플릿과 LLM을 체인으로 연결
explanation_template = PromptTemplate.from_template(
    """다음 주제에 대해 단계별로 설명해주세요:

주제: {topic}
대상: {audience}
형식: {format}

설명:"""
)

# 체인 생성 (LCEL 방식)
chain = explanation_template | llm

# 체인 실행
result = chain.invoke({
    "topic": "RAG(Retrieval-Augmented Generation)",
    "audience": "개발자",
    "format": "코드 예제 포함"
})

print(f"응답: {result.content[:300]}...")

# === 5. 다양한 템플릿 테스트 ===
print("\n📝 5. 다양한 템플릿 테스트")
print("-" * 30)

# 여러 템플릿 정의
templates = {
    "번역": PromptTemplate.from_template(
        "다음 {source_lang} 텍스트를 {target_lang}로 번역해주세요: '{text}'"
    ),
    "코드리뷰": PromptTemplate.from_template(
        "다음 {language} 코드를 리뷰하고 개선사항을 제안해주세요:\n\n{code}"
    ),
    "요약": PromptTemplate.from_template(
        "다음 텍스트를 {length}로 요약해주세요:\n\n{text}"
    ),
    "창작": PromptTemplate.from_template(
        "{genre} 장르의 {length} {type}을 '{theme}' 주제로 써주세요."
    )
}

# 각 템플릿 테스트
test_cases = [
    {
        "template": "번역",
        "params": {
            "source_lang": "영어",
            "target_lang": "한국어",
            "text": "Hello, how are you today?"
        }
    },
    {
        "template": "요약",
        "params": {
            "length": "3줄",
            "text": "인공지능은 컴퓨터가 인간처럼 학습하고 추론할 수 있도록 하는 기술입니다. 머신러닝, 딥러닝, 자연어처리 등 다양한 분야로 구성되어 있으며, 현재 많은 산업에서 활용되고 있습니다."
        }
    },
    {
        "template": "창작",
        "params": {
            "genre": "SF",
            "length": "짧은",
            "type": "이야기",
            "theme": "AI와 인간의 공존"
        }
    }
]

for i, test_case in enumerate(test_cases, 1):
    template_name = test_case["template"]
    params = test_case["params"]
    
    print(f"\n🧪 테스트 {i}: {template_name}")
    
    # 템플릿 포맷팅
    formatted_prompt = templates[template_name].format(**params)
    print(f"프롬프트: {formatted_prompt}")
    
    # Gemini로 실행
    try:
        response = llm.invoke(formatted_prompt)
        print(f"응답: {response.content[:150]}...")
    except Exception as e:
        print(f"오류: {e}")

# === 6. 대화형 템플릿 생성기 ===
print("\n📝 6. 대화형 템플릿 생성기")
print("-" * 30)
print("직접 템플릿을 만들어보세요! (종료: 'quit')")

while True:
    user_template = input("\n템플릿을 입력하세요 (예: '{name}님, {topic}에 대해 설명해주세요'): ").strip()
    
    if user_template.lower() in ['quit', 'exit', '종료']:
        break
    
    if not user_template:
        continue
    
    try:
        # 사용자 템플릿으로 PromptTemplate 생성
        custom_template = PromptTemplate.from_template(user_template)
        
        # 템플릿의 변수들 확인
        variables = custom_template.input_variables
        print(f"감지된 변수들: {variables}")
        
        # 각 변수의 값 입력받기
        params = {}
        for var in variables:
            value = input(f"{var} 값을 입력하세요: ").strip()
            params[var] = value
        
        # 템플릿 포맷팅
        formatted = custom_template.format(**params)
        print(f"\n생성된 프롬프트: {formatted}")
        
        # Gemini로 실행
        response = llm.invoke(formatted)
        print(f"Gemini 응답: {response.content}")
        
    except Exception as e:
        print(f"오류 발생: {e}")

print("\n🎉 PromptTemplate 예제가 완료되었습니다!")