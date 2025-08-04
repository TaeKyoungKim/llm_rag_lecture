# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# 필요한 라이브러리 임포트
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Gemini 모델 설정
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
output_parser = StrOutputParser()

# --- 체인 정의 (한 번만 생성) ---

# 1. 키워드 추출 체인
keyword_prompt = ChatPromptTemplate.from_messages([
    ("system", "다음 텍스트에서 핵심 키워드 3-5개를 추출해주세요. 키워드만 쉼표로 구분하여 반환하세요."),
    ("human", "{text}")
])
keyword_chain = keyword_prompt | llm | output_parser

# 2. 감정 분석 체인
sentiment_prompt = ChatPromptTemplate.from_messages([
    ("system", "다음 텍스트의 감정을 분석하고 '긍정적', '부정적', '중립적' 중 하나로 분류해주세요. 분류 결과만 반환하세요."),
    ("human", "{text}")
])
sentiment_chain = sentiment_prompt | llm | output_parser

# 3. 요약 체인
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "다음 텍스트를 2-3문장으로 요약해주세요."),
    ("human", "{text}")
])
summary_chain = summary_prompt | llm | output_parser

# 키워드 추출 함수
def extract_keywords(text):
    """텍스트에서 키워드를 추출하는 함수"""
    result = keyword_chain.invoke({"text": text})
    keywords = [keyword.strip() for keyword in result.split(',')]
    return keywords

# 감정 분석 함수  
def extract_sentiment(text):
    """텍스트의 감정을 분석하는 함수"""
    return sentiment_chain.invoke({"text": text}).strip()

# 요약 함수
def extract_summary(text):
    """텍스트를 요약하는 함수"""
    return summary_chain.invoke({"text": text})

# RunnableLambda로 함수들을 래핑
keyword_runnable = RunnableLambda(extract_keywords)
sentiment_runnable = RunnableLambda(extract_sentiment)
summary_runnable = RunnableLambda(extract_summary)

# 병렬 체인 생성 - 3가지 작업을 동시에 실행
parallel_chain = RunnableParallel({
    "keywords": keyword_runnable,
    "sentiment": sentiment_runnable,
    "summary": summary_runnable
})

# 테스트 함수
def analyze_text(input_text):
    """텍스트 분석 실행 함수"""
    print(f"분석할 텍스트: {input_text}\n")
    print("분석 중...")
    
    # 병렬로 키워드, 감정, 요약 추출
    result = parallel_chain.invoke(input_text)
    
    print("=" * 50)
    print("📊 분석 결과")
    print("=" * 50)
    print(f"🔑 키워드: {', '.join(result['keywords'])}")
    print(f"😊 감정: {result['sentiment']}")
    print(f"📝 요약: {result['summary']}")
    print("=" * 50)
    
    return result

# 사용 예시
if __name__ == "__main__":
    # 테스트 텍스트들
    test_texts = [
        "오늘 새로운 프로젝트를 시작했는데 정말 흥미진진합니다. 팀원들과 함께 협력하여 혁신적인 AI 솔루션을 개발할 예정입니다.",
        
        "회사에서 발표한 실적이 기대에 못 미쳐서 실망스럽습니다. 다음 분기에는 더 나은 결과를 위해 전략을 수정해야 할 것 같습니다.",
        
        "파이썬은 프로그래밍 언어 중 하나로, 간단하고 읽기 쉬운 문법을 가지고 있어 초보자들이 배우기 좋습니다. 데이터 과학, 웹 개발, 인공지능 등 다양한 분야에서 활용되고 있습니다."
    ]
    
    # 각 텍스트 분석
    for i, text in enumerate(test_texts, 1):
        print(f"\n🔍 테스트 {i}")
        try:
            analyze_text(text)
        except Exception as e:
            print(f"오류 발생: {e}")
        
        if i < len(test_texts):
            print("\n" + "="*80 + "\n")
    
    # 사용자 입력 받기
    print("\n" + "="*80)
    print("직접 텍스트를 입력해서 분석해보세요!")
    print("종료하려면 'quit' 입력")
    print("="*80)
    
    while True:
        user_input = input("\n분석할 텍스트를 입력하세요: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '종료', 'q']:
            print("프로그램을 종료합니다.")
            break
            
        if user_input:
            try:
                analyze_text(user_input)
            except Exception as e:
                print(f"오류 발생: {e}")
        else:
            print("텍스트를 입력해주세요.")