# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# 필요한 라이브러리 임포트
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

print("🔗 Gemini API LCEL Chain 구성 예제")
print("="*60)

# === 1. 기본 LCEL Chain ===
print("\n📝 1. 기본 LCEL Chain")
print("-" * 30)

# 컴포넌트 준비
prompt = ChatPromptTemplate.from_template("{text}")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
output_parser = StrOutputParser()

# LCEL로 체인 구성
basic_chain = prompt | llm | output_parser

# 체인 실행
result = basic_chain.invoke({"text": "안녕하세요!"})
print(f"입력: 안녕하세요!")
print(f"출력: {result}")

# === 2. 복합 프롬프트 Chain ===
print("\n📝 2. 복합 프롬프트 Chain")
print("-" * 30)

# 더 복잡한 프롬프트 템플릿
complex_prompt = ChatPromptTemplate.from_template(
    """주제: {topic}
난이도: {level}
형식: {format}

위 조건에 맞춰 설명해주세요."""
)

# 복합 체인 구성
complex_chain = complex_prompt | llm | output_parser

# 체인 실행
complex_result = complex_chain.invoke({
    "topic": "머신러닝",
    "level": "초보자",
    "format": "단계별 설명"
})

print(f"결과: {complex_result[:200]}...")

# === 3. 시스템 메시지가 포함된 Chain ===
print("\n📝 3. 시스템 메시지가 포함된 Chain")
print("-" * 30)

# 시스템 메시지 + 사용자 메시지
system_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role} 전문가입니다. {style} 스타일로 답변해주세요."),
    ("human", "{question}")
])

# 시스템 프롬프트 체인
system_chain = system_prompt | llm | output_parser

# 실행
system_result = system_chain.invoke({
    "role": "파이썬 프로그래밍",
    "style": "친절하고 상세한",
    "question": "리스트 컴프리헨션에 대해 설명해주세요."
})

print(f"결과: {system_result[:200]}...")

# === 4. 다단계 처리 Chain ===
print("\n📝 4. 다단계 처리 Chain")
print("-" * 30)

# 첫 번째 단계: 키워드 추출
keyword_prompt = ChatPromptTemplate.from_template(
    "다음 텍스트에서 핵심 키워드 3개를 추출해주세요. 키워드만 쉼표로 구분해서 답변하세요: {text}"
)

# 두 번째 단계: 키워드 기반 설명
explanation_prompt = ChatPromptTemplate.from_template(
    "다음 키워드들에 대해 종합적으로 설명해주세요: {keywords}"
)

# 다단계 체인 구성
multi_stage_chain = (
    {"text": RunnablePassthrough()} |
    keyword_prompt |
    llm |
    output_parser |
    {"keywords": RunnablePassthrough()} |
    explanation_prompt |
    llm |
    output_parser
)

# 실행
multi_result = multi_stage_chain.invoke("인공지능과 머신러닝은 현대 기술의 핵심이며, 데이터 과학과 밀접한 관련이 있습니다.")
print(f"최종 결과: {multi_result[:200]}...")

# === 5. 병렬 처리 Chain ===
print("\n📝 5. 병렬 처리 Chain")
print("-" * 30)

from langchain_core.runnables import RunnableParallel

# 병렬로 실행할 체인들
translation_prompt = ChatPromptTemplate.from_template("다음을 영어로 번역해주세요: {text}")
summary_prompt = ChatPromptTemplate.from_template("다음을 한 문장으로 요약해주세요: {text}")
sentiment_prompt = ChatPromptTemplate.from_template("다음 텍스트의 감정을 분석해주세요: {text}")

# 병렬 체인 구성
parallel_chain = RunnableParallel({
    "translation": translation_prompt | llm | output_parser,
    "summary": summary_prompt | llm | output_parser,
    "sentiment": sentiment_prompt | llm | output_parser
})

# 병렬 실행
parallel_result = parallel_chain.invoke({
    "text": "오늘은 정말 좋은 날씨입니다. 새로운 프로젝트를 시작하게 되어 매우 기대됩니다."
})

print("병렬 처리 결과:")
for key, value in parallel_result.items():
    print(f"- {key}: {value}")

# === 6. 조건부 처리 Chain ===
print("\n📝 6. 조건부 처리 Chain")
print("-" * 30)

# 텍스트 길이에 따른 조건부 처리
def choose_prompt(input_dict):
    text = input_dict["text"]
    if len(text) > 100:
        return ChatPromptTemplate.from_template("다음 긴 텍스트를 요약해주세요: {text}")
    else:
        return ChatPromptTemplate.from_template("다음 텍스트를 확장해서 설명해주세요: {text}")

# 조건부 체인
conditional_chain = (
    RunnableLambda(lambda x: {"text": x["text"], "prompt": choose_prompt(x)}) |
    RunnableLambda(lambda x: x["prompt"].format(text=x["text"])) |
    llm |
    output_parser
)

# 짧은 텍스트 테스트
short_text = "AI는 미래다."
conditional_result1 = conditional_chain.invoke({"text": short_text})
print(f"짧은 텍스트 결과: {conditional_result1[:150]}...")

# 긴 텍스트 테스트
long_text = "인공지능은 현대 사회의 많은 분야에서 혁신을 이끌고 있습니다. 머신러닝, 딥러닝, 자연어처리 등의 기술을 통해 우리는 이전에 불가능했던 많은 일들을 할 수 있게 되었습니다."
conditional_result2 = conditional_chain.invoke({"text": long_text})
print(f"긴 텍스트 결과: {conditional_result2[:150]}...")

# === 7. 커스텀 함수와 Chain 결합 ===
print("\n📝 7. 커스텀 함수와 Chain 결합")
print("-" * 30)

# 커스텀 전처리 함수
def preprocess_text(text):
    """텍스트 전처리 함수"""
    processed = text.strip().upper()
    return f"[전처리됨] {processed}"

# 커스텀 후처리 함수
def postprocess_result(result):
    """결과 후처리 함수"""
    return f"🤖 Gemini 답변: {result}\n📊 답변 길이: {len(result)}자"

# 전처리 + LLM + 후처리 체인
custom_chain = (
    RunnableLambda(preprocess_text) |
    ChatPromptTemplate.from_template("다음 텍스트에 대해 설명해주세요: {text}") |
    llm |
    output_parser |
    RunnableLambda(postprocess_result)
)

# 실행
custom_result = custom_chain.invoke("langchain은 무엇인가요?")
print(custom_result)

# === 8. 대화형 Chain 테스트 ===
print("\n📝 8. 대화형 Chain 테스트")
print("-" * 30)
print("직접 체인을 테스트해보세요! (종료: 'quit')")

# 다양한 체인 옵션
chain_options = {
    "1": {"name": "기본 질문답변", "chain": basic_chain},
    "2": {"name": "전문가 답변", "chain": system_chain},
    "3": {"name": "병렬 분석", "chain": parallel_chain}
}

while True:
    print("\n사용 가능한 체인:")
    for key, value in chain_options.items():
        print(f"{key}. {value['name']}")
    
    choice = input("\n체인을 선택하세요 (1-3, 또는 'quit'): ").strip()
    
    if choice.lower() in ['quit', 'exit', '종료']:
        break
    
    if choice not in chain_options:
        print("올바른 번호를 선택해주세요.")
        continue
    
    user_input = input("입력 텍스트: ").strip()
    if not user_input:
        continue
    
    try:
        selected_chain = chain_options[choice]["chain"]
        
        if choice == "2":  # 전문가 답변 체인
            result = selected_chain.invoke({
                "role": "AI 전문가",
                "style": "친근한",
                "question": user_input
            })
        else:  # 기본 체인들
            result = selected_chain.invoke({"text": user_input})
        
        print(f"\n✅ 결과:")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"- {key}: {value}")
        else:
            print(result)
            
    except Exception as e:
        print(f"❌ 오류: {e}")

print("\n🎉 LCEL Chain 예제가 완료되었습니다!")