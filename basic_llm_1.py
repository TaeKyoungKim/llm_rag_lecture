# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()
import os 
# 프롬프트, LLM, 출력 파서 연결
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# 시스템 메시지와 사용자 입력을 포함한 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 위트가 있고 장난기가 많은 AI 어시스턴트입니다. 항상 농담을 섞으면서도 정확한 답변을 제공해주세요."),
    ("human", "{input}")
])

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 항상 정중하고 정확한 답변을 제공해주세요."),
#     ("human", "{input}")
# ])

def run_chain(input_text):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0, max_tokens=1000)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain.invoke({"input": input_text})

if __name__ == "__main__":
    result = run_chain("안녕하세요!")
    print(result)