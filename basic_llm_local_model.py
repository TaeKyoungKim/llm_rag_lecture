from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser

# HuggingFace Model ID
model_id = 'beomi/llama-2-ko-7b'

# HuggingFacePipeline 객체 생성
llm = HuggingFacePipeline.from_model_id(
    model_id=model_id, 
    device=0,               # -1: CPU(default), 0번 부터는 CUDA 디바이스 번호 지정시 GPU 사용하여 추론
    task="text-generation", # 텍스트 생성
    model_kwargs={"temperature": 0.1, 
                  "max_length": 64},
)

# 템플릿
template = """질문: {question}

답변: """

# 프롬프트 템플릿 생성
prompt = PromptTemplate.from_template(template)

# 출력 파서
parser = StrOutputParser()

# LCEL을 사용한 체인 구성
chain = prompt | llm | parser

if __name__ == "__main__":
    question = "대한민국의 수도는 어디야?"
    result = chain.invoke({"question": question})
    print(result)