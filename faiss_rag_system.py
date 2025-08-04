"""
FAISS RAG 시스템
FAISS 검색 결과를 Gemini 2.5 Flash 모델로 답변을 생성하는 시스템
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# FAISS 라이브러리
import faiss

# LangChain 관련 라이브러리
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

class FAISSRAGSystem:
    """
    FAISS RAG 시스템 클래스
    FAISS 검색 결과를 Gemini 2.5 Flash 모델로 답변 생성
    """
    
    def __init__(self, embedding_type: str = "huggingface", prompt_style: str = "detailed"):
        """
        RAG 시스템 초기화
        Args:
            embedding_type: "huggingface" 또는 "gemini"
            prompt_style: "detailed", "simple", "academic" 중 선택
        """
        print("🤖 FAISS RAG 시스템 초기화")
        
        # 파일 경로 설정
        self.index_dir = Path("DocumentsLoader/educational_faiss_index")
        
        # 임베딩 타입 설정
        self.embedding_type = embedding_type
        
        # 프롬프트 스타일 설정
        self.prompt_style = prompt_style
        
        # HNSW 설정
        self.hnsw_config = {
            'M': 16,
            'efConstruction': 100,
            'efSearch': 50,
            'metric': faiss.METRIC_INNER_PRODUCT
        }
        
        # 임베딩 모델 초기화
        self._initialize_embeddings()
        
        # LLM 모델 초기화
        self._initialize_llm()
        
        # FAISS 인덱스 초기화
        self.faiss_index = None
        
        # 프롬프트 템플릿 초기화
        self._initialize_prompts()
        
        print("   ✅ RAG 시스템 초기화 완료")
    
    def _initialize_embeddings(self):
        """임베딩 모델 초기화"""
        print("   🔄 임베딩 모델 초기화 중...")
        
        if self.embedding_type == "gemini":
            try:
                # Gemini API 키 확인
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    print("   ⚠️ GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
                    print("   💡 HuggingFace 임베딩으로 자동 전환합니다.")
                    self._load_huggingface_embeddings()
                    return
                
                print("   🔑 Gemini API 키 확인됨")
                
                # Gemini 임베딩 모델 초기화
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=api_key,
                    task_type="retrieval_query",
                    title="Technical Analysis Document"
                )
                
                # 간단한 테스트로 연결 확인
                test_embedding = self.embeddings.embed_query("test")
                if test_embedding and len(test_embedding) > 0:
                    print("   ✅ Gemini 임베딩 모델 로드 및 테스트 완료")
                else:
                    raise Exception("임베딩 테스트 실패")
                    
            except Exception as e:
                error_msg = str(e)
                print(f"   ⚠️ Gemini 임베딩 로드 실패: {error_msg}")
                
                # 오류 유형별 안내 메시지
                if "invalid_grant" in error_msg or "Bad Request" in error_msg:
                    print("   💡 API 키 인증 오류입니다. 다음을 확인해주세요:")
                    print("      - GOOGLE_API_KEY가 올바른지 확인")
                    print("      - API 키가 Gemini API에 대한 권한이 있는지 확인")
                    print("      - API 키가 만료되지 않았는지 확인")
                elif "timeout" in error_msg.lower():
                    print("   💡 네트워크 타임아웃입니다. 인터넷 연결을 확인해주세요.")
                elif "quota" in error_msg.lower():
                    print("   💡 API 할당량 초과입니다. 잠시 후 다시 시도하거나 다른 API 키를 사용하세요.")
                else:
                    print("   💡 알 수 없는 오류입니다. HuggingFace 임베딩으로 대체합니다.")
                
                print("   🔄 HuggingFace 임베딩으로 자동 전환 중...")
                self._load_huggingface_embeddings()
        else:
            self._load_huggingface_embeddings()
    
    def _load_huggingface_embeddings(self):
        """HuggingFace 임베딩 모델 로드"""
        try:
            print("   🔄 HuggingFace 임베딩 모델 로드 중...")
            
            # 여러 모델 옵션 제공
            model_options = [
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            ]
            
            for model_name in model_options:
                try:
                    print(f"   🔍 모델 시도 중: {model_name}")
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                    
                    # 간단한 테스트
                    test_embedding = self.embeddings.embed_query("test")
                    if test_embedding and len(test_embedding) > 0:
                        print(f"   ✅ HuggingFace 임베딩 모델 로드 완료: {model_name}")
                        self.embedding_type = "huggingface"
                        return
                    else:
                        raise Exception("임베딩 테스트 실패")
                        
                except Exception as e:
                    print(f"   ⚠️ {model_name} 로드 실패: {str(e)}")
                    continue
            
            # 모든 모델이 실패한 경우
            raise Exception("사용 가능한 HuggingFace 모델을 찾을 수 없습니다.")
            
        except Exception as e:
            print(f"   ❌ HuggingFace 임베딩 로드 실패: {str(e)}")
            print("   💡 오프라인 모드로 전환하거나 인터넷 연결을 확인해주세요.")
            raise
    
    def _initialize_llm(self):
        """LLM 모델 초기화"""
        print("   🔄 LLM 모델 초기화 중...")
        
        try:
            # Gemini API 키 확인
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise Exception("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
            
            print("   🔑 Gemini API 키 확인됨")
            
            # Gemini 2.5 Flash 모델 초기화
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=api_key,
                temperature=0.7,
                max_output_tokens=2048,
                convert_system_message_to_human=True
            )
            
            # 간단한 테스트
            test_response = self.llm.invoke("안녕하세요. 간단한 테스트입니다.")
            if test_response and test_response.content:
                print("   ✅ Gemini 2.5 Flash LLM 모델 로드 및 테스트 완료")
            else:
                raise Exception("LLM 테스트 실패")
                
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ LLM 모델 로드 실패: {error_msg}")
            
            # 오류 유형별 안내 메시지
            if "invalid_grant" in error_msg or "Bad Request" in error_msg:
                print("   💡 API 키 인증 오류입니다. 다음을 확인해주세요:")
                print("      - GOOGLE_API_KEY가 올바른지 확인")
                print("      - API 키가 Gemini API에 대한 권한이 있는지 확인")
                print("      - API 키가 만료되지 않았는지 확인")
            elif "timeout" in error_msg.lower():
                print("   💡 네트워크 타임아웃입니다. 인터넷 연결을 확인해주세요.")
            elif "quota" in error_msg.lower():
                print("   💡 API 할당량 초과입니다. 잠시 후 다시 시도하거나 다른 API 키를 사용하세요.")
            else:
                print("   💡 알 수 없는 오류입니다. API 키와 인터넷 연결을 확인해주세요.")
            
            raise
    
    def _initialize_prompts(self):
        """프롬프트 템플릿 초기화"""
        print(f"   🔄 프롬프트 템플릿 초기화 중... (스타일: {self.prompt_style})")
        
        # 프롬프트 스타일별 템플릿 정의
        prompt_templates = {
            "detailed": ChatPromptTemplate.from_messages([
                ("system", """당신은 기술적 분석 전문가입니다. 주어진 문서를 바탕으로 질문에 대해 정확하고 유용한 답변을 제공해주세요.

답변 시 다음 사항을 준수해주세요:
1. 제공된 문서 내용을 기반으로 답변하세요
2. 문서에 없는 내용은 추측하지 마세요
3. 명확하고 이해하기 쉽게 설명하세요
4. 필요시 예시나 구체적인 설명을 포함하세요
5. 한국어로 답변하세요
6. 답변의 시작에 "문서 내용을 바탕으로 답변드리겠습니다."라는 문구를 포함하세요
7. 답변의 끝에 "이상이 문서에서 찾을 수 있는 내용입니다."라는 문구로 마무리하세요
8. 만약 문서에서 관련 내용을 찾을 수 없다면, "죄송합니다. 제공된 문서에서 해당 질문과 관련된 내용을 찾을 수 없습니다."라고 답변하세요
9. 답변은 구조화하여 작성하세요 (예: 주요 개념, 특징, 활용법 등)"""),
                ("human", """다음 문서 내용을 참고하여 질문에 답변해주세요:

문서 내용:
{context}

질문: {question}

위 문서 내용을 바탕으로 질문에 대한 답변을 제공해주세요.""")
            ]),
            
            "simple": ChatPromptTemplate.from_messages([
                ("system", """당신은 기술적 분석 전문가입니다. 주어진 문서를 바탕으로 질문에 대해 간결하고 명확한 답변을 제공해주세요.

답변 시 다음 사항을 준수해주세요:
1. 제공된 문서 내용만을 기반으로 답변하세요
2. 간결하고 핵심적인 내용만 포함하세요
3. 한국어로 답변하세요
4. 불필요한 형식적 문구는 생략하세요"""),
                ("human", """문서 내용:
{context}

질문: {question}

답변해주세요.""")
            ]),
            
            "academic": ChatPromptTemplate.from_messages([
                ("system", """당신은 금융공학 및 기술적 분석 분야의 학술 전문가입니다. 주어진 문서를 바탕으로 학술적이고 체계적인 답변을 제공해주세요.

답변 시 다음 사항을 준수해주세요:
1. 제공된 문서 내용을 기반으로 학술적 관점에서 답변하세요
2. 개념적 정의, 수학적 원리, 실무 적용 순서로 구성하세요
3. 전문 용어를 적절히 사용하되, 이해하기 쉽게 설명하세요
4. 한국어로 답변하세요
5. 답변의 시작에 "학술적 관점에서 답변드리겠습니다."라는 문구를 포함하세요
6. 답변의 끝에 "이상이 학술적 분석 결과입니다."라는 문구로 마무리하세요"""),
                ("human", """다음 문서 내용을 참고하여 학술적 관점에서 질문에 답변해주세요:

문서 내용:
{context}

질문: {question}

학술적이고 체계적인 답변을 제공해주세요.""")
            ])
        }
        
        # 선택된 스타일의 프롬프트 템플릿 사용
        if self.prompt_style in prompt_templates:
            self.prompt_template = prompt_templates[self.prompt_style]
        else:
            print(f"   ⚠️ 알 수 없는 프롬프트 스타일 '{self.prompt_style}'. 기본 스타일(detailed)을 사용합니다.")
            self.prompt_template = prompt_templates["detailed"]
        
        # LLM 체인 생성 (최신 LangChain 방식)
        self.llm_chain = self.prompt_template | self.llm | StrOutputParser()
        
        print(f"   ✅ ChatPromptTemplate 초기화 완료 (스타일: {self.prompt_style})")
    
    def load_index(self) -> bool:
        """저장된 FAISS 인덱스 로드"""
        try:
            if not (self.index_dir / "index.faiss").exists():
                print(f"   ❌ FAISS 인덱스 파일을 찾을 수 없습니다: {self.index_dir}")
                print("   💡 먼저 'faiss_index_builder.py'를 실행하여 인덱스를 구축해주세요.")
                return False
            
            print("📂 FAISS HNSW 인덱스 로드 중...")
            self.faiss_index = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True  # 로컬에서 생성한 파일이므로 안전
            )
            print("   ✅ FAISS 인덱스 로드 완료")
            
            # 인덱스 정보 출력
            self._print_index_info()
            
            return True
            
        except Exception as e:
            print(f"   ❌ 인덱스 로드 실패: {str(e)}")
            return False
    
    def _print_index_info(self):
        """인덱스 정보 출력"""
        if not self.faiss_index:
            return
        
        print("\n📊 인덱스 정보:")
        print("-" * 30)
        
        # 기본 정보
        print(f"   • 임베딩 모델: {self.embedding_type}")
        print(f"   • LLM 모델: Gemini 2.5 Flash")
        print(f"   • 프롬프트 스타일: {self.prompt_style}")
        print(f"   • 인덱스 타입: FAISS HNSW")
        print(f"   • 로드 위치: {self.index_dir}")
        
        # HNSW 정보
        if hasattr(self.faiss_index.index, 'hnsw'):
            print(f"   • HNSW 노드 수: {self.faiss_index.index.hnsw.levels.size()}")
            print(f"   • HNSW 최대 레벨: {self.faiss_index.index.hnsw.max_level}")
            print(f"   • HNSW efSearch: {self.faiss_index.index.hnsw.efSearch}")
        
        # 문서 정보
        docs = list(self.faiss_index.docstore.values())
        if docs:
            technical_docs = sum(1 for doc in docs if doc.metadata.get('has_technical_content', False))
            print(f"   • 총 문서 수: {len(docs)}")
            print(f"   • 기술적 내용 문서: {technical_docs}")
            print(f"   • 평균 문서 길이: {sum(len(doc.page_content) for doc in docs) // len(docs)}자")
    
    def search_and_generate_answer(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        검색 및 답변 생성
        Args:
            query: 검색 쿼리
            k: 검색할 문서 수
        Returns:
            검색 결과와 생성된 답변을 포함한 딕셔너리
        """
        if not self.faiss_index:
            print("❌ FAISS 인덱스가 로드되지 않았습니다.")
            return {}
        
        try:
            print(f"\n🔍 질문: '{query}'")
            print("=" * 60)
            
            # HNSW 검색 파라미터 설정
            if hasattr(self.faiss_index.index, 'hnsw'):
                self.faiss_index.index.hnsw.efSearch = self.hnsw_config['efSearch']
                print(f"   🔧 HNSW efSearch 설정: {self.hnsw_config['efSearch']}")
            
            # 1단계: 유사도 검색
            print("   🔄 1단계: 유사도 검색 수행 중...")
            
            # 직접 FAISS 인덱스를 사용하여 검색
            try:
                # 쿼리 임베딩 생성
                query_embedding = self.embeddings.embed_query(query)
                query_vector = np.array([query_embedding], dtype=np.float32)
                
                # FAISS 검색 수행
                distances, indices = self.faiss_index.index.search(query_vector, k)
                
                # 결과 구성
                docs_and_scores = []
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx != -1:  # 유효한 인덱스인 경우
                        doc = self.faiss_index.docstore[idx]
                        # 거리를 유사도 점수로 변환 (코사인 유사도 기준)
                        similarity_score = 1.0 - distance
                        docs_and_scores.append((doc, similarity_score))
                
                print(f"   ✅ {len(docs_and_scores)}개 관련 문서 발견")
                
            except Exception as search_error:
                print(f"   ⚠️ 직접 검색 실패: {str(search_error)}")
                print("   🔄 LangChain 래퍼로 재시도...")
                
                # LangChain 래퍼로 재시도
                docs_and_scores = self.faiss_index.similarity_search_with_score(
                    query, k=k
                )
            
            if not docs_and_scores:
                print("   ❌ 관련 문서를 찾을 수 없습니다.")
                return {
                    'query': query,
                    'search_results': [],
                    'answer': "죄송합니다. 질문과 관련된 문서를 찾을 수 없습니다.",
                    'context': "",
                    'sources': []
                }
            
            # 2단계: 컨텍스트 구성
            print("   🔄 2단계: 컨텍스트 구성 중...")
            
            # 검색 결과 출력
            print(f"\n📄 검색된 문서 ({len(docs_and_scores)}개):")
            print("-" * 40)
            
            context_parts = []
            sources = []
            
            for i, (doc, score) in enumerate(docs_and_scores, 1):
                print(f"   📄 문서 {i}:")
                print(f"      - 유사도 점수: {score:.4f}")
                print(f"      - 페이지: {doc.metadata.get('page', 'N/A')}")
                print(f"      - 청크 크기: {doc.metadata.get('chunk_size', 'N/A')}자")
                print(f"      - 기술적 내용: {'✅' if doc.metadata.get('has_technical_content', False) else '❌'}")
                print(f"      - 내용 미리보기: {doc.page_content[:100]}...")
                print()
                
                # 컨텍스트에 추가
                context_parts.append(f"[문서 {i}] {doc.page_content}")
                sources.append({
                    'page': doc.metadata.get('page', 'N/A'),
                    'chunk_id': doc.metadata.get('chunk_id', 'N/A'),
                    'score': float(score),
                    'content_preview': doc.page_content[:200] + "..."
                })
            
            # 전체 컨텍스트 구성
            context = "\n\n".join(context_parts)
            print(f"   ✅ 컨텍스트 구성 완료 ({len(context)}자)")
            
            # 3단계: 답변 생성
            print("   🔄 3단계: Gemini 2.5 Flash로 답변 생성 중...")
            
            try:
                # LLM 체인으로 답변 생성 (최신 LangChain 방식)
                response = self.llm_chain.invoke({
                    "context": context,
                    "question": query
                })
                
                answer = str(response)
                print("   ✅ 답변 생성 완료")
                
            except Exception as llm_error:
                print(f"   ❌ 답변 생성 실패: {str(llm_error)}")
                answer = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
            
            # 결과 구성
            result = {
                'query': query,
                'search_results': docs_and_scores,
                'answer': answer,
                'context': context,
                'sources': sources,
                'embedding_type': self.embedding_type,
                'llm_model': 'gemini-2.5-flash',
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"   ❌ 검색 및 답변 생성 실패: {str(e)}")
            return {
                'query': query,
                'search_results': [],
                'answer': f"오류가 발생했습니다: {str(e)}",
                'context': "",
                'sources': []
            }
    
    def save_rag_results(self, result: Dict[str, Any]):
        """RAG 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rag_results_{timestamp}.json"
        
        try:
            # JSON 직렬화 가능한 형태로 변환
            save_result = {
                'query': result.get('query', ''),
                'answer': result.get('answer', ''),
                'embedding_type': result.get('embedding_type', ''),
                'llm_model': result.get('llm_model', ''),
                'timestamp': result.get('timestamp', ''),
                'sources': result.get('sources', []),
                'context_preview': result.get('context', '')[:1000] + "..." if result.get('context') else ""
            }
            
            output_path = Path("DocumentsLoader/rag_results") / filename
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_result, f, ensure_ascii=False, indent=2)
            
            print(f"   ✅ RAG 결과 저장: {output_path}")
            
        except Exception as e:
            print(f"   ❌ 결과 저장 실패: {str(e)}")
    
    def interactive_rag(self):
        """대화형 RAG 모드"""
        print("\n🤖 대화형 RAG 모드")
        print("=" * 60)
        print("사용 가능한 명령어:")
        print("  - 질문만 입력: 직접 질문 (예: 'RSI란 무엇인가요?', '볼린저밴드 사용법')")
        print("  - 'ask <질문>': 명시적 질문 명령")
        print("  - 'info': 시스템 정보")
        print("  - 'quit': 종료")
        print(f"\n현재 모델:")
        print(f"  • 임베딩: {self.embedding_type}")
        print(f"  • LLM: Gemini 2.5 Flash")
        print(f"  • 프롬프트 스타일: {self.prompt_style}")
        print()
        
        while True:
            try:
                command = input("질문 입력: ").strip()
                
                if command.lower() == 'quit':
                    break
                elif command.lower() == 'info':
                    self._print_index_info()
                elif command.startswith('ask '):
                    # 명시적 ask 명령어 처리
                    question = command[4:].strip()
                    if question:
                        result = self.search_and_generate_answer(question)
                        if result and result.get('answer'):
                            print(f"\n🤖 답변:")
                            print("-" * 40)
                            print(result['answer'])
                            print()
                            self.save_rag_results(result)
                    else:
                        print("   ❌ 질문을 입력해주세요.")
                elif command:
                    # 단순 질문으로 처리 (ask 접두사 없이)
                    result = self.search_and_generate_answer(command)
                    if result and result.get('answer'):
                        print(f"\n🤖 답변:")
                        print("-" * 40)
                        print(result['answer'])
                        print()
                        self.save_rag_results(result)
                    else:
                        print("   ❌ 답변을 생성할 수 없습니다.")
                else:
                    print("   ❌ 질문을 입력해주세요.")
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 RAG 시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")

def main():
    """메인 실행 함수"""
    import sys
    
    # 명령행 인수 처리
    embedding_type = "huggingface"
    prompt_style = "detailed"
    
    if len(sys.argv) > 1:
        embedding_type = sys.argv[1].lower()
    
    if len(sys.argv) > 2:
        prompt_style = sys.argv[2].lower()
    
    # 유효성 검사
    if embedding_type not in ["huggingface", "gemini"]:
        print("❌ 지원하지 않는 임베딩 타입입니다. 'huggingface' 또는 'gemini'를 사용하세요.")
        return
    
    if prompt_style not in ["detailed", "simple", "academic"]:
        print("❌ 지원하지 않는 프롬프트 스타일입니다. 'detailed', 'simple', 'academic' 중 선택하세요.")
        return
    
    print(f"🤖 FAISS RAG 시스템")
    print(f"   • 임베딩 모델: {embedding_type}")
    print(f"   • 프롬프트 스타일: {prompt_style}")
    
    # RAG 시스템 초기화
    rag_system = FAISSRAGSystem(embedding_type=embedding_type, prompt_style=prompt_style)
    
    # 인덱스 로드
    if rag_system.load_index():
        print("\n🎉 RAG 시스템 준비 완료!")
        
        # 대화형 RAG 모드 실행
        rag_system.interactive_rag()
        
    else:
        print("❌ RAG 시스템 초기화에 실패했습니다.")
        print("💡 먼저 'faiss_index_builder.py'를 실행하여 인덱스를 구축해주세요.")

if __name__ == "__main__":
    main() 