# 간단한 FAISS HNSW 테스트 시스템 (HNSW 명시적 구현 + Gemini 임베딩 지원)

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import faiss

# LangChain 관련 임포트
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

print("🔧 간단한 FAISS HNSW 테스트 시스템 (HNSW 명시적 구현 + Gemini 임베딩)")
print("=" * 70)

class SimpleFAISSTest:
    """간단한 FAISS 테스트 시스템 (HNSW 명시적 구현 + Gemini 임베딩 지원)"""
    
    def __init__(self, embedding_type: str = "huggingface"):
        """
        초기화
        Args:
            embedding_type: "huggingface" 또는 "gemini"
        """
        self.index_dir = Path("DocumentsLoader/faiss_index")
        self.index_dir.mkdir(exist_ok=True)
        
        # 텍스트 분할 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # 임베딩 모델 설정
        self.embedding_type = embedding_type
        if embedding_type == "gemini":
            try:
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                print("   ✅ Gemini 임베딩 모델 로드 완료")
            except Exception as e:
                print(f"   ⚠️ Gemini 임베딩 로드 실패, HuggingFace로 대체: {str(e)}")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                self.embedding_type = "huggingface"
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("   ✅ HuggingFace 임베딩 모델 로드 완료")
        
        self.faiss_index = None
        
        # HNSW 설정
        self.hnsw_config = {
            'M': 16,  # 각 노드의 최대 연결 수
            'efConstruction': 200,  # 구축 시 탐색할 이웃 수
            'efSearch': 50,  # 검색 시 탐색할 이웃 수
            'metric': faiss.METRIC_INNER_PRODUCT  # 코사인 유사도
        }
    
    def create_sample_documents(self) -> List[Document]:
        """샘플 기술적 분석 문서 생성"""
        print("📝 샘플 문서 생성 중...")
        
        sample_texts = [
            "RSI(Relative Strength Index)는 상대강도지수로, 주가의 상승폭과 하락폭을 비교하여 과매수/과매도 상태를 판단하는 지표입니다. 일반적으로 70 이상이면 과매수, 30 이하면 과매도로 판단합니다.",
            
            "MACD(Moving Average Convergence Divergence)는 이동평균수렴확산지수로, 두 개의 이동평균선의 차이를 이용한 추세 추종형 지표입니다. 골든크로스와 데드크로스로 매매 시점을 판단합니다.",
            
            "볼린저밴드는 주가의 변동성을 측정하는 지표로, 중심선(이동평균)과 상하한선(표준편차)으로 구성됩니다. 밴드가 좁아지면 큰 움직임이 예상되고, 밴드가 넓어지면 변동성이 커집니다.",
            
            "이동평균선은 일정 기간의 주가 평균을 연결한 선으로, 추세를 파악하는 기본적인 지표입니다. 단기, 중기, 장기 이동평균선의 배열로 매매 시점을 판단할 수 있습니다.",
            
            "스토캐스틱은 주가가 일정 기간의 고가와 저가 범위 내에서 어느 위치에 있는지를 나타내는 오실레이터입니다. %K와 %D 두 선으로 구성되며, 과매수/과매도 구간을 판단합니다.",
            
            "일목균형표는 일본의 일목산인이 개발한 기술적 분석 도구로, 시간론, 가격론, 파동론, 형보론의 네 가지 요소로 구성됩니다. 구름대(일목균형표)를 통해 지지/저항을 판단합니다.",
            
            "피보나치 되돌림은 피보나치 수열을 이용한 기술적 분석 도구로, 주가의 상승이나 하락 후 되돌림의 깊이를 예측합니다. 주요 되돌림 레벨은 23.6%, 38.2%, 50%, 61.8%입니다.",
            
            "엘리어트 파동이론은 주가의 움직임이 일정한 패턴을 반복한다는 이론으로, 5개의 상승파와 3개의 하락파로 구성됩니다. 파동의 특성을 이해하면 시장의 전환점을 예측할 수 있습니다.",
            
            "지지선과 저항선은 주가가 상승하거나 하락할 때 멈추는 가격대를 의미합니다. 지지선은 주가가 하락할 때 지지받는 가격대이고, 저항선은 주가가 상승할 때 저항받는 가격대입니다.",
            
            "거래량은 주식의 거래 활성도를 나타내는 지표로, 주가와 함께 분석하면 시장의 강약을 판단할 수 있습니다. 거래량이 증가하면서 주가가 상승하면 강세 신호로 해석됩니다."
        ]
        
        documents = []
        for i, text in enumerate(sample_texts):
            doc = Document(
                page_content=text,
                metadata={
                    'id': i + 1,
                    'type': 'technical_analysis',
                    'language': 'ko',
                    'embedding_type': self.embedding_type,
                    'created_time': datetime.now().isoformat()
                }
            )
            documents.append(doc)
        
        print(f"   ✅ {len(documents)}개 샘플 문서 생성 완료")
        return documents
    
    def create_faiss_index(self, documents: List[Document]) -> bool:
        """FAISS HNSW 인덱스 생성 (명시적 HNSW 구현)"""
        print("🔍 FAISS HNSW 인덱스 생성 중...")
        print(f"   📊 임베딩 모델: {self.embedding_type}")
        print(f"   🔧 HNSW 설정: M={self.hnsw_config['M']}, efConstruction={self.hnsw_config['efConstruction']}")
        
        try:
            # 문서 분할
            split_docs = self.text_splitter.split_documents(documents)
            print(f"   📊 {len(split_docs)}개 청크로 분할 완료")
            
            # 임베딩 생성
            print("   🔄 임베딩 생성 중...")
            embeddings_list = []
            for doc in split_docs:
                embedding = self.embeddings.embed_query(doc.page_content)
                embeddings_list.append(embedding)
            
            # FAISS 인덱스 생성 (HNSW 명시적 구현)
            dimension = len(embeddings_list[0])
            print(f"   📏 임베딩 차원: {dimension}")
            
            # HNSW 인덱스 생성
            index = faiss.IndexHNSWFlat(dimension, self.hnsw_config['M'])
            index.hnsw.efConstruction = self.hnsw_config['efConstruction']
            index.hnsw.efSearch = self.hnsw_config['efSearch']
            index.metric_type = self.hnsw_config['metric']
            
            # 벡터 추가
            embeddings_array = faiss.vector_to_array(embeddings_list)
            index.add(embeddings_array)
            
            # LangChain FAISS 래퍼 생성
            self.faiss_index = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=self._create_docstore(split_docs),
                index_to_docstore_id={i: i for i in range(len(split_docs))}
            )
            
            # 인덱스 저장
            self.faiss_index.save_local(str(self.index_dir))
            
            print(f"   ✅ FAISS HNSW 인덱스 생성 완료")
            print(f"   📁 저장 위치: {self.index_dir}")
            print(f"   📊 총 벡터 수: {index.ntotal}")
            print(f"   🔧 HNSW 노드 수: {index.hnsw.levels.size()}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 인덱스 생성 실패: {str(e)}")
            return False
    
    def _create_docstore(self, documents: List[Document]):
        """문서 저장소 생성"""
        from langchain.docstore.document import Document as LangChainDocument
        
        docstore = {}
        for i, doc in enumerate(documents):
            docstore[i] = LangChainDocument(
                page_content=doc.page_content,
                metadata=doc.metadata
            )
        return docstore
    
    def load_index(self) -> bool:
        """기존 인덱스 로드"""
        try:
            if (self.index_dir / "index.faiss").exists():
                print("📂 기존 FAISS HNSW 인덱스 로드 중...")
                self.faiss_index = FAISS.load_local(
                    str(self.index_dir),
                    self.embeddings
                )
                print("   ✅ 인덱스 로드 완료")
                
                # HNSW 정보 출력
                if hasattr(self.faiss_index.index, 'hnsw'):
                    print(f"   🔧 HNSW 노드 수: {self.faiss_index.index.hnsw.levels.size()}")
                    print(f"   🔧 HNSW 최대 레벨: {self.faiss_index.index.hnsw.max_level}")
                
                return True
            else:
                print("   ⚠️ 기존 인덱스가 없습니다.")
                return False
        except Exception as e:
            print(f"   ❌ 인덱스 로드 실패: {str(e)}")
            return False
    
    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """검색 수행 (HNSW 최적화)"""
        if not self.faiss_index:
            print("❌ 인덱스가 로드되지 않았습니다.")
            return []
        
        try:
            print(f"🔍 HNSW 검색: '{query}'")
            
            # HNSW 검색 파라미터 설정
            if hasattr(self.faiss_index.index, 'hnsw'):
                self.faiss_index.index.hnsw.efSearch = self.hnsw_config['efSearch']
            
            # 유사도 검색
            results = self.faiss_index.similarity_search_with_score(query, k=k)
            
            print(f"   ✅ {len(results)}개 결과 발견")
            
            # 결과 출력
            for i, (doc, score) in enumerate(results, 1):
                print(f"   📄 결과 {i}:")
                print(f"      - 유사도 점수: {score:.4f}")
                print(f"      - 문서 ID: {doc.metadata.get('id', 'N/A')}")
                print(f"      - 임베딩 타입: {doc.metadata.get('embedding_type', 'N/A')}")
                print(f"      - 내용: {doc.page_content[:100]}...")
                print()
            
            return results
            
        except Exception as e:
            print(f"   ❌ 검색 실패: {str(e)}")
            return []
    
    def test_searches(self):
        """다양한 검색 테스트"""
        print("\n🧪 HNSW 검색 테스트 시작")
        print("=" * 50)
        
        # 1. RSI 검색
        print("\n1. RSI 검색:")
        self.search("RSI 상대강도지수")
        
        # 2. MACD 검색
        print("\n2. MACD 검색:")
        self.search("MACD 이동평균수렴확산")
        
        # 3. 볼린저밴드 검색
        print("\n3. 볼린저밴드 검색:")
        self.search("볼린저밴드 변동성")
        
        # 4. 이동평균선 검색
        print("\n4. 이동평균선 검색:")
        self.search("이동평균선 추세")
        
        # 5. 스토캐스틱 검색
        print("\n5. 스토캐스틱 검색:")
        self.search("스토캐스틱 오실레이터")
        
        # 6. 일목균형표 검색
        print("\n6. 일목균형표 검색:")
        self.search("일목균형표 구름대")
        
        # 7. 피보나치 검색
        print("\n7. 피보나치 검색:")
        self.search("피보나치 되돌림 레벨")
        
        # 8. 엘리어트 파동 검색
        print("\n8. 엘리어트 파동 검색:")
        self.search("엘리어트 파동이론")
        
        # 9. 지지저항 검색
        print("\n9. 지지저항 검색:")
        self.search("지지선 저항선")
        
        # 10. 거래량 검색
        print("\n10. 거래량 검색:")
        self.search("거래량 분석")
    
    def get_index_info(self) -> Dict[str, Any]:
        """인덱스 정보 조회"""
        if not self.faiss_index:
            return {}
        
        try:
            info = {
                'embedding_type': self.embedding_type,
                'total_vectors': len(self.faiss_index.docstore._dict),
                'index_type': 'FAISS HNSW',
                'hnsw_config': self.hnsw_config,
                'index_path': str(self.index_dir)
            }
            
            # HNSW 특정 정보
            if hasattr(self.faiss_index.index, 'hnsw'):
                info.update({
                    'hnsw_nodes': self.faiss_index.index.hnsw.levels.size(),
                    'hnsw_max_level': self.faiss_index.index.hnsw.max_level,
                    'hnsw_ef_search': self.faiss_index.index.hnsw.efSearch,
                    'hnsw_ef_construction': self.faiss_index.index.hnsw.efConstruction
                })
            
            return info
            
        except Exception as e:
            print(f"❌ 인덱스 정보 조회 실패: {str(e)}")
            return {}
    
    def run(self):
        """메인 실행 함수"""
        print("🚀 FAISS HNSW 테스트 시스템 시작")
        print(f"📊 임베딩 모델: {self.embedding_type}")
        
        # 1. 기존 인덱스 로드 시도
        if not self.load_index():
            # 2. 인덱스가 없으면 새로 생성
            print("\n📝 새 HNSW 인덱스 생성 시작")
            documents = self.create_sample_documents()
            if not self.create_faiss_index(documents):
                print("❌ 인덱스 생성에 실패했습니다.")
                return
        
        # 3. 인덱스 정보 출력
        info = self.get_index_info()
        print(f"\n📊 인덱스 정보:")
        for key, value in info.items():
            print(f"   • {key}: {value}")
        
        # 4. 검색 테스트 수행
        self.test_searches()
        
        print("\n🎉 FAISS HNSW 테스트 완료!")

def main():
    """메인 함수"""
    import sys
    
    # 명령행 인수로 임베딩 타입 선택
    embedding_type = "huggingface"
    if len(sys.argv) > 1:
        embedding_type = sys.argv[1].lower()
    
    if embedding_type not in ["huggingface", "gemini"]:
        print("❌ 지원하지 않는 임베딩 타입입니다. 'huggingface' 또는 'gemini'를 사용하세요.")
        return
    
    print(f"🔧 사용할 임베딩 모델: {embedding_type}")
    
    test_system = SimpleFAISSTest(embedding_type=embedding_type)
    test_system.run()

if __name__ == "__main__":
    main() 