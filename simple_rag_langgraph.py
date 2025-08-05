"""
SimpleRAG with LangGraph
FAISSSearchEngine을 임포트하여 LangGraph로 구현한 RAG 시스템
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, TypedDict, Annotated
import numpy as np

# LangGraph 관련 라이브러리
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# LangChain 관련 라이브러리
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# FAISS 검색 엔진 임포트
from faiss_search_engine import FAISSSearchEngine

# 그래프 시각화 라이브러리
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch, ConnectionPatch
    import networkx as nx
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    print("⚠️ 그래프 시각화를 위해 matplotlib과 networkx를 설치하세요: uv add matplotlib networkx")

from dotenv import load_dotenv
load_dotenv()

# 상태 정의
class RAGState(TypedDict):
    """RAG 시스템의 상태를 정의하는 클래스"""
    query: str
    search_results: List[Tuple[Document, float]]
    context: str
    answer: str
    sources: List[Dict[str, Any]]
    error: Optional[str]
    metadata: Dict[str, Any]

class SimpleRAGWithLangGraph:
    """
    LangGraph를 사용한 SimpleRAG 시스템
    FAISSSearchEngine을 임포트하여 검색과 답변 생성을 분리된 노드로 구성
    """
    
    def __init__(self, embedding_type: str = "huggingface", prompt_style: str = "simple"):
        """
        SimpleRAG 시스템 초기화
        Args:
            embedding_type: "huggingface" 또는 "gemini"
            prompt_style: "simple", "detailed", "academic" 중 선택
        """
        print("🤖 SimpleRAG with LangGraph 초기화")
        
        # 설정 저장
        self.embedding_type = embedding_type
        self.prompt_style = prompt_style
        
        # FAISS 검색 엔진 초기화
        self.search_engine = FAISSSearchEngine(embedding_type=embedding_type)
        
        # LLM 초기화
        self._initialize_llm()
        
        # 프롬프트 템플릿 초기화
        self._initialize_prompts()
        
        # LangGraph 워크플로우 구성
        self.workflow = self._create_workflow()
        
        print("   ✅ SimpleRAG 시스템 초기화 완료")
    
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
            print(f"   ⚠️ 알 수 없는 프롬프트 스타일 '{self.prompt_style}'. 기본 스타일(simple)을 사용합니다.")
            self.prompt_template = prompt_templates["simple"]
        
        # LLM 체인 생성
        self.llm_chain = self.prompt_template | self.llm | StrOutputParser()
        
        print(f"   ✅ ChatPromptTemplate 초기화 완료 (스타일: {self.prompt_style})")
    
    def _create_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 생성"""
        print("   🔄 LangGraph 워크플로우 구성 중...")
        
        # 상태 그래프 생성
        workflow = StateGraph(RAGState)
        
        # 노드 추가
        workflow.add_node("search", self._search_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # 엣지 설정
        workflow.set_entry_point("search")
        workflow.add_edge("search", "generate_answer")
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("handle_error", END)
        
        # 조건부 엣지 설정
        workflow.add_conditional_edges(
            "search",
            self._should_generate_answer,
            {
                "continue": "generate_answer",
                "error": "handle_error"
            }
        )
        
        # 워크플로우 컴파일
        compiled_workflow = workflow.compile(checkpointer=MemorySaver())
        
        print("   ✅ LangGraph 워크플로우 구성 완료")
        return compiled_workflow
    
    def visualize_workflow(self, save_path: str = "simple_rag_workflow.png"):
        """워크플로우 시각화"""
        if not GRAPH_AVAILABLE:
            print("❌ 그래프 시각화를 위해 matplotlib과 networkx를 설치하세요:")
            print("   uv add matplotlib networkx")
            return
        
        try:
            print("🎨 워크플로우 시각화 생성 중...")
            
            # 그래프 생성
            G = nx.DiGraph()
            
            # 노드 추가
            nodes = [
                ("입력 쿼리", {"pos": (0, 2), "color": "#E3F2FD", "type": "input"}),
                ("검색 노드", {"pos": (2, 2), "color": "#FFF3E0", "type": "process"}),
                ("조건부 분기", {"pos": (4, 2), "color": "#F3E5F5", "type": "decision"}),
                ("답변 생성 노드", {"pos": (6, 3), "color": "#E8F5E8", "type": "process"}),
                ("오류 처리 노드", {"pos": (6, 1), "color": "#FFEBEE", "type": "error"}),
                ("최종 답변", {"pos": (8, 2), "color": "#E0F2F1", "type": "output"})
            ]
            
            for node, attrs in nodes:
                G.add_node(node, **attrs)
            
            # 엣지 추가
            edges = [
                ("입력 쿼리", "검색 노드", "쿼리 전달"),
                ("검색 노드", "조건부 분기", "검색 결과"),
                ("조건부 분기", "답변 생성 노드", "성공"),
                ("조건부 분기", "오류 처리 노드", "실패"),
                ("답변 생성 노드", "최종 답변", "생성된 답변"),
                ("오류 처리 노드", "최종 답변", "오류 메시지")
            ]
            
            for edge in edges:
                G.add_edge(edge[0], edge[1], label=edge[2])
            
            # 그래프 그리기
            plt.figure(figsize=(14, 8))
            ax = plt.gca()
            
            # 노드 위치
            pos = nx.get_node_attributes(G, 'pos')
            
            # 노드 그리기
            for node, (x, y) in pos.items():
                node_attrs = G.nodes[node]
                color = node_attrs['color']
                node_type = node_attrs['type']
                
                # 노드 모양 결정
                if node_type == "input":
                    shape = "s"  # 사각형
                    size = 3000
                elif node_type == "output":
                    shape = "s"  # 사각형
                    size = 3000
                elif node_type == "decision":
                    shape = "d"  # 다이아몬드
                    size = 4000
                elif node_type == "error":
                    shape = "o"  # 원
                    size = 2500
                else:
                    shape = "o"  # 원
                    size = 3000
                
                nx.draw_networkx_nodes(G, pos, nodelist=[node], 
                                     node_color=color, node_size=size, 
                                     node_shape=shape, ax=ax)
            
            # 엣지 그리기
            nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                 arrows=True, arrowsize=20, 
                                 arrowstyle='->', ax=ax)
            
            # 노드 라벨
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            
            # 엣지 라벨
            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
            
            # 제목 및 설정
            plt.title("SimpleRAG LangGraph 워크플로우", fontsize=16, fontweight='bold', pad=20)
            plt.axis('off')
            
            # 범례 추가
            legend_elements = [
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#E3F2FD', 
                          markersize=15, label='입력/출력'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFF3E0', 
                          markersize=15, label='처리 노드'),
                plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='#F3E5F5', 
                          markersize=15, label='조건부 분기'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFEBEE', 
                          markersize=15, label='오류 처리')
            ]
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
            
            # 시스템 정보 텍스트 추가
            info_text = f"""
시스템 정보:
• 임베딩 모델: {self.embedding_type}
• 프롬프트 스타일: {self.prompt_style}
• 워크플로우 엔진: LangGraph
• 노드 구성: search → generate_answer
            """
            plt.figtext(0.02, 0.02, info_text, fontsize=9, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            # 저장
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"   ✅ 워크플로우 시각화 저장: {save_path}")
            
        except Exception as e:
            print(f"   ❌ 워크플로우 시각화 실패: {str(e)}")
    
    def _search_node(self, state: RAGState) -> RAGState:
        """검색 노드 - FAISS 검색 수행"""
        try:
            print(f"🔍 검색 노드 실행: '{state['query']}'")
            
            # FAISS 검색 수행
            search_results = self.search_engine.search(state['query'], k=5)
            
            if search_results:
                # 컨텍스트 구성
                context_parts = []
                sources = []
                
                for i, (doc, score) in enumerate(search_results, 1):
                    context_parts.append(f"[문서 {i}] {doc.page_content}")
                    sources.append({
                        'page': doc.metadata.get('page', 'N/A'),
                        'chunk_id': doc.metadata.get('chunk_id', 'N/A'),
                        'score': float(score),
                        'content_preview': doc.page_content[:200] + "..."
                    })
                
                context = "\n\n".join(context_parts)
                
                return {
                    **state,
                    'search_results': search_results,
                    'context': context,
                    'sources': sources,
                    'error': None
                }
            else:
                return {
                    **state,
                    'search_results': [],
                    'context': "",
                    'sources': [],
                    'error': "관련 문서를 찾을 수 없습니다."
                }
                
        except Exception as e:
            return {
                **state,
                'error': f"검색 중 오류 발생: {str(e)}"
            }
    
    def _generate_answer_node(self, state: RAGState) -> RAGState:
        """답변 생성 노드 - LLM으로 답변 생성"""
        try:
            print("🤖 답변 생성 노드 실행")
            
            if not state['context']:
                answer = "죄송합니다. 질문과 관련된 문서를 찾을 수 없습니다."
            else:
                # LLM으로 답변 생성
                response = self.llm_chain.invoke({
                    "context": state['context'],
                    "question": state['query']
                })
                
                answer = str(response)
            
            return {
                **state,
                'answer': answer,
                'metadata': {
                    'embedding_type': self.embedding_type,
                    'llm_model': 'gemini-2.5-flash',
                    'prompt_style': self.prompt_style,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {
                **state,
                'error': f"답변 생성 중 오류 발생: {str(e)}",
                'answer': "죄송합니다. 답변 생성 중 오류가 발생했습니다."
            }
    
    def _handle_error_node(self, state: RAGState) -> RAGState:
        """오류 처리 노드"""
        print(f"❌ 오류 처리 노드 실행: {state.get('error', '알 수 없는 오류')}")
        
        return {
            **state,
            'answer': f"오류가 발생했습니다: {state.get('error', '알 수 없는 오류')}",
            'context': "",
            'sources': []
        }
    
    def _should_generate_answer(self, state: RAGState) -> str:
        """검색 후 답변 생성 여부 결정"""
        if state.get('error'):
            return "error"
        return "continue"
    
    def load_index(self) -> bool:
        """FAISS 인덱스 로드"""
        return self.search_engine.load_index()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        쿼리 처리 - LangGraph 워크플로우 실행
        Args:
            query: 사용자 질문
        Returns:
            처리 결과 딕셔너리
        """
        try:
            print(f"\n🤖 SimpleRAG 처리 시작: '{query}'")
            print("=" * 60)
            
            # 초기 상태 설정
            initial_state = RAGState(
                query=query,
                search_results=[],
                context="",
                answer="",
                sources=[],
                error=None,
                metadata={}
            )
            
            # 워크플로우 실행
            result = self.workflow.invoke(initial_state)
            
            # 결과 출력
            print(f"\n🤖 답변:")
            print("-" * 40)
            print(result['answer'])
            
            if result['sources']:
                print(f"\n📄 참고 문서 ({len(result['sources'])}개):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"   {i}. 페이지 {source['page']} (유사도: {source['score']:.4f})")
            
            return result
            
        except Exception as e:
            print(f"❌ 쿼리 처리 실패: {str(e)}")
            return {
                'query': query,
                'answer': f"오류가 발생했습니다: {str(e)}",
                'context': "",
                'sources': [],
                'error': str(e)
            }
    
    def save_results(self, result: Dict[str, Any]):
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_rag_results_{timestamp}.json"
        
        try:
            # JSON 직렬화 가능한 형태로 변환
            save_result = {
                'query': result.get('query', ''),
                'answer': result.get('answer', ''),
                'embedding_type': result.get('metadata', {}).get('embedding_type', ''),
                'llm_model': result.get('metadata', {}).get('llm_model', ''),
                'prompt_style': result.get('metadata', {}).get('prompt_style', ''),
                'timestamp': result.get('metadata', {}).get('timestamp', ''),
                'sources': result.get('sources', []),
                'context_preview': result.get('context', '')[:1000] + "..." if result.get('context') else ""
            }
            
            output_path = Path("DocumentsLoader/simple_rag_results") / filename
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_result, f, ensure_ascii=False, indent=2)
            
            print(f"   ✅ 결과 저장: {output_path}")
            
        except Exception as e:
            print(f"   ❌ 결과 저장 실패: {str(e)}")
    
    def interactive_mode(self):
        """대화형 모드"""
        print("\n🤖 SimpleRAG 대화형 모드")
        print("=" * 60)
        print("사용 가능한 명령어:")
        print("  - 질문만 입력: 직접 질문 (예: 'RSI란 무엇인가요?', '볼린저밴드 사용법')")
        print("  - 'ask <질문>': 명시적 질문 명령")
        print("  - 'info': 시스템 정보")
        print("  - 'graph': 워크플로우 시각화")
        print("  - 'quit': 종료")
        print(f"\n현재 설정:")
        print(f"  • 임베딩: {self.embedding_type}")
        print(f"  • LLM: Gemini 2.5 Flash")
        print(f"  • 프롬프트 스타일: {self.prompt_style}")
        print(f"  • 워크플로우: LangGraph")
        print()
        
        while True:
            try:
                command = input("질문 입력: ").strip()
                
                if command.lower() == 'quit':
                    break
                elif command.lower() == 'info':
                    print(f"\n📊 시스템 정보:")
                    print(f"  • 임베딩 모델: {self.embedding_type}")
                    print(f"  • LLM 모델: Gemini 2.5 Flash")
                    print(f"  • 프롬프트 스타일: {self.prompt_style}")
                    print(f"  • 워크플로우 엔진: LangGraph")
                    print(f"  • 노드 구성: search → generate_answer")
                elif command.lower() == 'graph':
                    self.visualize_workflow()
                elif command.startswith('ask '):
                    question = command[4:].strip()
                    if question:
                        result = self.process_query(question)
                        self.save_results(result)
                    else:
                        print("   ❌ 질문을 입력해주세요.")
                elif command:
                    result = self.process_query(command)
                    self.save_results(result)
                else:
                    print("   ❌ 질문을 입력해주세요.")
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 SimpleRAG 시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")

def main():
    """메인 실행 함수"""
    import sys
    
    # 명령행 인수 처리
    embedding_type = "huggingface"
    prompt_style = "simple"
    
    if len(sys.argv) > 1:
        embedding_type = sys.argv[1].lower()
    
    if len(sys.argv) > 2:
        prompt_style = sys.argv[2].lower()
    
    # 유효성 검사
    if embedding_type not in ["huggingface", "gemini"]:
        print("❌ 지원하지 않는 임베딩 타입입니다. 'huggingface' 또는 'gemini'를 사용하세요.")
        return
    
    if prompt_style not in ["simple", "detailed", "academic"]:
        print("❌ 지원하지 않는 프롬프트 스타일입니다. 'simple', 'detailed', 'academic' 중 선택하세요.")
        return
    
    print(f"🤖 SimpleRAG with LangGraph")
    print(f"   • 임베딩 모델: {embedding_type}")
    print(f"   • 프롬프트 스타일: {prompt_style}")
    print(f"   • 워크플로우 엔진: LangGraph")
    
    # SimpleRAG 시스템 초기화
    simple_rag = SimpleRAGWithLangGraph(embedding_type=embedding_type, prompt_style=prompt_style)
    
    # 인덱스 로드
    if simple_rag.load_index():
        print("\n🎉 SimpleRAG 시스템 준비 완료!")
        
        # 워크플로우 시각화 생성
        simple_rag.visualize_workflow()
        
        # 대화형 모드 실행
        simple_rag.interactive_mode()
        
    else:
        print("❌ SimpleRAG 시스템 초기화에 실패했습니다.")
        print("💡 먼저 'faiss_index_builder.py'를 실행하여 인덱스를 구축해주세요.")

if __name__ == "__main__":
    main() 