"""
FAISS HNSW 검색 엔진
저장된 FAISS 인덱스를 로드하여 유사도 검색을 수행하는 시스템
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
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from dotenv import load_dotenv
load_dotenv()

class FAISSSearchEngine:
    """
    FAISS HNSW 검색 엔진 클래스
    저장된 FAISS 인덱스를 로드하여 유사도 검색 수행
    """
    
    def __init__(self, embedding_type: str = "huggingface"):
        """
        검색 엔진 초기화
        Args:
            embedding_type: "huggingface" 또는 "gemini"
        """
        print("🔍 FAISS 검색 엔진 초기화")
        
        # 파일 경로 설정
        self.index_dir = Path("DocumentsLoader/educational_faiss_index")
        
        # 임베딩 타입 설정
        self.embedding_type = embedding_type
        
        # HNSW 설정
        self.hnsw_config = {
            'M': 16,
            'efConstruction': 100,
            'efSearch': 50,
            'metric': faiss.METRIC_INNER_PRODUCT
        }
        
        # 임베딩 모델 초기화
        self._initialize_embeddings()
        
        # FAISS 인덱스 초기화
        self.faiss_index = None
        
        print("   ✅ 검색 엔진 초기화 완료")
    
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
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        유사도 검색 수행
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
        Returns:
            검색 결과 리스트 (문서, 유사도 점수)
        """
        if not self.faiss_index:
            print("❌ FAISS 인덱스가 로드되지 않았습니다.")
            return []
        
        try:
            print(f"\n🔍 검색 수행: '{query}'")
            print("-" * 40)
            
            # HNSW 검색 파라미터 설정
            if hasattr(self.faiss_index.index, 'hnsw'):
                self.faiss_index.index.hnsw.efSearch = self.hnsw_config['efSearch']
                print(f"   🔧 HNSW efSearch 설정: {self.hnsw_config['efSearch']}")
            
            # 유사도 검색 수행
            print("   🔄 검색 수행 중...")
            
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
                
                print(f"   ✅ {len(docs_and_scores)}개 결과 발견")
                
            except Exception as search_error:
                print(f"   ⚠️ 직접 검색 실패: {str(search_error)}")
                print("   🔄 LangChain 래퍼로 재시도...")
                
                # LangChain 래퍼로 재시도
                docs_and_scores = self.faiss_index.similarity_search_with_score(
                    query, k=k
                )
            
            print(f"   ✅ {len(docs_and_scores)}개 결과 발견")
            
            # 결과 출력
            for i, (doc, score) in enumerate(docs_and_scores, 1):
                print(f"\n   📄 결과 {i}:")
                print(f"      - 유사도 점수: {score:.4f}")
                print(f"      - 페이지: {doc.metadata.get('page', 'N/A')}")
                print(f"      - 청크 크기: {doc.metadata.get('chunk_size', 'N/A')}자")
                print(f"      - 기술적 내용: {'✅' if doc.metadata.get('has_technical_content', False) else '❌'}")
                print(f"      - 내용 미리보기: {doc.page_content[:100]}...")
            
            return docs_and_scores
            
        except Exception as e:
            print(f"   ❌ 검색 실패: {str(e)}")
            return []
    
    def save_search_results(self, results: List[Tuple[Document, float]], query: str):
        """검색 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_results_{timestamp}.json"
        
        try:
            search_data = {
                'query': query,
                'embedding_type': self.embedding_type,
                'timestamp': datetime.now().isoformat(),
                'total_results': len(results),
                'results': []
            }
            
            for doc, score in results:
                result_item = {
                    'score': float(score),
                    'page': doc.metadata.get('page', 'N/A'),
                    'chunk_id': doc.metadata.get('chunk_id', 'N/A'),
                    'chunk_size': doc.metadata.get('chunk_size', 'N/A'),
                    'has_technical_content': doc.metadata.get('has_technical_content', False),
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                search_data['results'].append(result_item)
            
            output_path = Path("DocumentsLoader/search_results") / filename
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(search_data, f, ensure_ascii=False, indent=2)
            
            print(f"   ✅ 검색 결과 저장: {output_path}")
            
        except Exception as e:
            print(f"   ❌ 결과 저장 실패: {str(e)}")
    
    def interactive_search(self):
        """대화형 검색 모드"""
        print("\n🔍 대화형 검색 모드")
        print("=" * 50)
        print("사용 가능한 명령어:")
        print("  - 검색어만 입력: 직접 검색 (예: 'RSI', '볼린저밴드')")
        print("  - 'search <검색어>': 명시적 검색 명령")
        print("  - 'info': 인덱스 정보")
        print("  - 'quit': 종료")
        print(f"\n현재 임베딩 모델: {self.embedding_type}")
        print()
        
        while True:
            try:
                command = input("검색 명령어 입력: ").strip()
                
                if command.lower() == 'quit':
                    break
                elif command.lower() == 'info':
                    self._print_index_info()
                elif command.startswith('search '):
                    # 명시적 search 명령어 처리
                    query = command[7:].strip()
                    if query:
                        results = self.search(query)
                        if results:
                            self.save_search_results(results, query)
                    else:
                        print("   ❌ 검색어를 입력해주세요.")
                elif command:
                    # 단순 검색어로 처리 (search 접두사 없이)
                    results = self.search(command)
                    if results:
                        self.save_search_results(results, command)
                    else:
                        print("   ❌ 검색 결과가 없습니다.")
                else:
                    print("   ❌ 검색어를 입력해주세요.")
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 검색을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")
    
    def batch_search(self, queries: List[str], k: int = 5) -> Dict[str, List[Tuple[Document, float]]]:
        """
        배치 검색 수행
        Args:
            queries: 검색 쿼리 리스트
            k: 각 쿼리당 반환할 결과 수
        Returns:
            쿼리별 검색 결과 딕셔너리
        """
        if not self.faiss_index:
            print("❌ FAISS 인덱스가 로드되지 않았습니다.")
            return {}
        
        print(f"\n🔍 배치 검색 수행: {len(queries)}개 쿼리")
        print("-" * 40)
        
        results = {}
        for i, query in enumerate(queries, 1):
            print(f"   📝 쿼리 {i}/{len(queries)}: '{query}'")
            query_results = self.search(query, k)
            results[query] = query_results
            
            if query_results:
                print(f"      ✅ {len(query_results)}개 결과 발견")
            else:
                print(f"      ❌ 결과 없음")
        
        return results

def main():
    """메인 실행 함수"""
    import sys
    
    # 명령행 인수로 임베딩 타입 선택
    embedding_type = "huggingface"
    if len(sys.argv) > 1:
        embedding_type = sys.argv[1].lower()
    
    if embedding_type not in ["huggingface", "gemini"]:
        print("❌ 지원하지 않는 임베딩 타입입니다. 'huggingface' 또는 'gemini'를 사용하세요.")
        return
    
    print(f"🔍 FAISS 검색 엔진 - 임베딩 모델: {embedding_type}")
    
    # 검색 엔진 초기화
    search_engine = FAISSSearchEngine(embedding_type=embedding_type)
    
    # 인덱스 로드
    if search_engine.load_index():
        print("\n🎉 검색 엔진 준비 완료!")
        
        # 대화형 검색 모드 실행
        search_engine.interactive_search()
        
    else:
        print("❌ 검색 엔진 초기화에 실패했습니다.")
        print("💡 먼저 'faiss_index_builder.py'를 실행하여 인덱스를 구축해주세요.")

if __name__ == "__main__":
    main() 