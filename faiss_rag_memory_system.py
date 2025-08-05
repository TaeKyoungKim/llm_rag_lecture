"""
FAISS RAG 시스템 + 대화 메모리 통합
FAISS 문서 검색 + 대화 기록 기억 기능
"""

import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# FAISS 및 LangChain 라이브러리
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from operator import itemgetter

from dotenv import load_dotenv
load_dotenv()

class FAISSRAGWithMemory:
    """FAISS RAG + 대화 메모리 통합 시스템"""
    
    def __init__(self, 
                 faiss_index_dir: str = "DocumentsLoader/educational_faiss_index",
                 memory_persist_dir: str = "./conversation_memory",
                 embedding_type: str = "huggingface"):
        
        print("🤖 FAISS RAG + 메모리 시스템 초기화")
        
        self.faiss_index_dir = Path(faiss_index_dir)
        self.embedding_type = embedding_type
        
        # 임베딩 모델 초기화
        self._initialize_embeddings()
        
        # LLM 모델 초기화
        self._initialize_llm()
        
        # FAISS 인덱스 (문서 검색용)
        self.faiss_index = None
        
        # Chroma 벡터스토어 (대화 메모리용)
        self.memory_store = Chroma(
            persist_directory=memory_persist_dir,
            embedding_function=self.embeddings
        )
        
        # 현재 세션 관리
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_buffer = []
        
        # 체인 설정
        self._setup_chains()
        
        print("   ✅ 초기화 완료")
    
    def _initialize_embeddings(self):
        """임베딩 모델 초기화"""
        if self.embedding_type == "gemini":
            try:
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise Exception("GOOGLE_API_KEY가 설정되지 않음")
                
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=api_key
                )
                print("   ✅ Gemini 임베딩 로드 완료")
                
            except Exception as e:
                print(f"   ⚠️ Gemini 임베딩 실패: {e}")
                print("   🔄 HuggingFace로 전환")
                self._load_huggingface_embeddings()
        else:
            self._load_huggingface_embeddings()
    
    def _load_huggingface_embeddings(self):
        """HuggingFace 임베딩 로드"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.embedding_type = "huggingface"
        print("   ✅ HuggingFace 임베딩 로드 완료")
    
    def _initialize_llm(self):
        """LLM 초기화"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise Exception("GOOGLE_API_KEY 환경 변수가 필요합니다")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=2048,
            convert_system_message_to_human=True
        )
        print("   ✅ Gemini 2.0 Flash LLM 로드 완료")
    
    def _setup_chains(self):
        """RAG + 메모리 체인 구성"""
        
        def search_documents(query: str) -> str:
            """FAISS에서 문서 검색"""
            if not self.faiss_index or not hasattr(self.faiss_index, 'similarity_search_with_score'):
                return ""
            
            try:
                docs_and_scores = self.faiss_index.similarity_search_with_score(query, k=3)
                if docs_and_scores:
                    results = []
                    for doc, score in docs_and_scores:
                        results.append(f"[문서] {doc.page_content}")
                    return "\n\n".join(results)
            except Exception:
                # FAISS 인덱스 문제가 있어도 조용히 처리
                pass
            
            return ""
        
        def search_conversations(query: str) -> str:
            """메모리에서 대화 검색"""
            if not hasattr(self.memory_store, 'similarity_search'):
                return ""
                
            try:
                docs = self.memory_store.similarity_search(query, k=3)
                conversations = []
                for doc in docs:
                    if doc.metadata.get('type') == 'conversation':
                        conversations.append(f"[이전대화] {doc.page_content}")
                return "\n\n".join(conversations)
            except Exception:
                # 메모리가 비어있거나 문제가 있어도 조용히 처리
                return ""
        
        def get_recent_context(inputs) -> str:
            """최근 대화 컨텍스트"""
            if not self.conversation_buffer:
                return ""
            
            recent = self.conversation_buffer[-2:]  # 최근 2턴
            context = []
            for turn in recent:
                context.append(f"사용자: {turn['human']}")
                context.append(f"AI: {turn['ai']}")
            return "\n".join(context)
        
        # 병렬 검색 체인
        self.search_chain = RunnableParallel({
            "query": itemgetter("query"),
            "documents": itemgetter("query") | RunnableLambda(search_documents),
            "conversations": itemgetter("query") | RunnableLambda(search_conversations),
            "recent_context": RunnableLambda(get_recent_context)
        })
        
        # 프롬프트 템플릿
        self.prompt = ChatPromptTemplate.from_template("""
당신은 도움이 되는 AI 어시스턴트입니다.

검색된 문서:
{documents}

이전 대화 기록:
{conversations}

최근 대화:
{recent_context}

사용자 질문: {query}

답변 지침:
1. 문서 내용이 있으면 우선적으로 활용
2. 이전 대화가 있으면 일관성 유지
3. 최근 대화 맥락 고려
4. 한국어로 명확하게 답변

답변:
""")
        
        # 전체 체인
        self.rag_chain = (
            self.search_chain
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def load_faiss_index(self) -> bool:
        """FAISS 인덱스 로드"""
        try:
            if not (self.faiss_index_dir / "index.faiss").exists():
                print(f"❌ FAISS 인덱스 없음: {self.faiss_index_dir}")
                return False
            
            print("📂 FAISS 인덱스 로드 중...")
            self.faiss_index = FAISS.load_local(
                str(self.faiss_index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("   ✅ FAISS 인덱스 로드 완료")
            
            # 인덱스 정보
            docs = list(self.faiss_index.docstore.values())
            print(f"   📄 총 문서: {len(docs)}개")
            
            return True
            
        except Exception as e:
            print(f"❌ 인덱스 로드 실패: {e}")
            return False
    
    def add_conversation_to_memory(self, human_msg: str, ai_msg: str, 
                                  importance_threshold: float = 5.0):
        """대화를 메모리에 저장"""
        
        # 중요도 계산 (간단한 휴리스틱)
        importance = self._calculate_importance(human_msg, ai_msg)
        
        # 현재 세션 버퍼에 추가
        self.conversation_buffer.append({
            "human": human_msg,
            "ai": ai_msg,
            "importance": importance,
            "timestamp": datetime.now()
        })
        
        # 버퍼 크기 제한
        if len(self.conversation_buffer) > 10:
            self.conversation_buffer.pop(0)
        
        # 중요한 대화만 벡터 스토어에 저장
        if importance >= importance_threshold:
            conversation_text = f"사용자: {human_msg}\nAI: {ai_msg}"
            metadata = {
                "type": "conversation",
                "session_id": self.current_session_id,
                "importance": importance,
                "timestamp": datetime.now().isoformat()
            }
            
            self.memory_store.add_texts([conversation_text], [metadata])
            self.memory_store.persist()
            print(f"💾 대화 저장됨 (중요도: {importance:.1f})")
    
    def _calculate_importance(self, human_msg: str, ai_msg: str) -> float:
        """대화 중요도 계산"""
        score = 3.0  # 기본 점수
        
        # 길이 기반
        if len(human_msg) > 50:
            score += 1.0
        
        # 키워드 기반
        important_keywords = [
            "배우", "학습", "공부", "기억", "설명", "방법", "어떻게", 
            "무엇", "언제", "왜", "프로젝트", "계획", "중요"
        ]
        
        for keyword in important_keywords:
            if keyword in human_msg or keyword in ai_msg:
                score += 0.5
        
        return min(score, 10.0)
    
    def chat(self, message: str) -> Dict[str, Any]:
        """대화 처리"""
        print(f"\n🔍 질문: {message}")
        
        try:
            # RAG 체인으로 답변 생성
            response = self.rag_chain.invoke({"query": message})
            
            # 메모리에 저장
            self.add_conversation_to_memory(message, response)
            
            # 소스 정보 수집
            sources = self._get_sources(message)
            
            result = {
                "query": message,
                "answer": response,
                "sources": sources,
                "session_id": self.current_session_id,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 답변 생성 실패: {e}")
            return {
                "query": message,
                "answer": f"죄송합니다. 오류가 발생했습니다: {str(e)}",
                "sources": {"documents": [], "conversations": []},
                "session_id": self.current_session_id
            }
    
    def _get_sources(self, query: str) -> Dict[str, List]:
        """검색 소스 정보 수집"""
        sources = {"documents": [], "conversations": []}
        
        # 문서 소스 (FAISS)
        if self.faiss_index and hasattr(self.faiss_index, 'similarity_search_with_score'):
            try:
                docs_and_scores = self.faiss_index.similarity_search_with_score(query, k=3)
                for doc, score in docs_and_scores:
                    sources["documents"].append({
                        "content": doc.page_content[:200] + "...",
                        "score": float(score),
                        "metadata": doc.metadata
                    })
            except Exception as e:
                # FAISS 인덱스가 없어도 정상 동작하도록 조용히 처리
                pass
        
        # 대화 소스 (Chroma)
        try:
            if hasattr(self.memory_store, 'similarity_search'):
                conv_docs = self.memory_store.similarity_search(query, k=3)
                for doc in conv_docs:
                    if doc.metadata.get('type') == 'conversation':
                        sources["conversations"].append({
                            "content": doc.page_content[:200] + "...",
                            "metadata": doc.metadata
                        })
        except Exception as e:
            # 메모리가 비어있어도 정상 동작하도록 조용히 처리
            pass
        
        return sources
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계"""
        try:
            # 전체 메모리 항목 수
            all_docs = self.memory_store.similarity_search("", k=1000)
            conversations = [d for d in all_docs if d.metadata.get('type') == 'conversation']
            
            return {
                "total_memory_items": len(all_docs),
                "conversation_items": len(conversations),
                "current_buffer_size": len(self.conversation_buffer),
                "faiss_documents": len(list(self.faiss_index.docstore.values())) if self.faiss_index else 0,
                "current_session": self.current_session_id
            }
        except Exception as e:
            return {"error": str(e)}
    
    def search_memory(self, query: str, k: int = 5) -> List[Dict]:
        """메모리 검색"""
        try:
            docs = self.memory_store.similarity_search_with_score(query, k=k)
            results = []
            
            for doc, score in docs:
                results.append({
                    "content": doc.page_content,
                    "score": float(score),
                    "type": doc.metadata.get('type', 'unknown'),
                    "timestamp": doc.metadata.get('timestamp')
                })
            
            return results
        except Exception as e:
            print(f"메모리 검색 오류: {e}")
            return []
    
    def start_new_session(self) -> str:
        """새 세션 시작"""
        # 이전 세션 요약 저장 (옵션)
        if len(self.conversation_buffer) >= 3:
            self._save_session_summary()
        
        # 새 세션 시작
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_buffer = []
        
        print(f"🔄 새 세션 시작: {self.current_session_id}")
        return self.current_session_id
    
    def _save_session_summary(self):
        """세션 요약 저장"""
        try:
            full_conversation = "\n".join([
                f"사용자: {turn['human']}\nAI: {turn['ai']}"
                for turn in self.conversation_buffer
            ])
            
            summary_prompt = f"다음 대화를 간단히 요약해주세요:\n\n{full_conversation}"
            summary = self.llm.invoke(summary_prompt).content
            
            # 요약을 메모리에 저장
            metadata = {
                "type": "session_summary",
                "session_id": self.current_session_id,
                "timestamp": datetime.now().isoformat(),
                "original_turns": len(self.conversation_buffer)
            }
            
            self.memory_store.add_texts([f"세션 요약: {summary}"], [metadata])
            self.memory_store.persist()
            
            print("📝 세션 요약 저장됨")
            
        except Exception as e:
            print(f"요약 저장 오류: {e}")
    
    def interactive_mode(self):
        """대화형 모드"""
        print("\n🤖 대화형 FAISS RAG + 메모리 시스템")
        print("명령어: 'stats' (통계), 'search <질문>' (메모리 검색), 'new' (새 세션), 'quit' (종료)")
        print()
        
        while True:
            try:
                user_input = input("질문: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'stats':
                    stats = self.get_memory_stats()
                    print("\n📊 시스템 통계:")
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
                    print()
                elif user_input.lower() == 'new':
                    self.start_new_session()
                elif user_input.startswith('search '):
                    query = user_input[7:].strip()
                    results = self.search_memory(query)
                    print(f"\n🔍 메모리 검색 결과 ({len(results)}개):")
                    for i, result in enumerate(results, 1):
                        print(f"   {i}. [{result['type']}] {result['content'][:100]}...")
                    print()
                elif user_input:
                    result = self.chat(user_input)
                    print(f"\n🤖 답변:")
                    print("-" * 40)
                    print(result['answer'])
                    
                    # 소스 정보 출력
                    sources = result['sources']
                    if sources['documents']:
                        print(f"\n📄 참조 문서: {len(sources['documents'])}개")
                    if sources['conversations']:
                        print(f"💭 관련 대화: {len(sources['conversations'])}개")
                    print()
                
            except KeyboardInterrupt:
                print("\n👋 시스템 종료")
                break
            except Exception as e:
                print(f"오류: {e}")

def main():
    """메인 실행 함수"""
    import sys
    
    # 명령행 인수 처리
    embedding_type = "huggingface"
    if len(sys.argv) > 1:
        embedding_type = sys.argv[1].lower()
    
    print(f"🤖 FAISS RAG + 메모리 시스템 (임베딩: {embedding_type})")
    
    try:
        # 시스템 초기화
        rag_memory = FAISSRAGWithMemory(embedding_type=embedding_type)
        
        # FAISS 인덱스 로드
        if rag_memory.load_faiss_index():
            print("🎉 시스템 준비 완료!")
            
            # 대화형 모드 실행
            rag_memory.interactive_mode()
        else:
            print("❌ FAISS 인덱스 로드 실패")
            print("💡 먼저 인덱스를 구축해주세요.")
            
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")

if __name__ == "__main__":
    main()