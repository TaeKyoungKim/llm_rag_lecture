# 완전한 FAISS HNSW 시스템 - 실제 PDF 데이터 활용 (HNSW 명시적 구현 + Gemini 임베딩)

import os
import json
import re
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

# 기존 PDF 처리 시스템 임포트
from process_technical_analysis_pdf_improved import ImprovedPDFProcessor

print("🔧 완전한 FAISS HNSW 시스템 초기화 (HNSW 명시적 구현 + Gemini 임베딩)")
print("=" * 80)

class CompleteFAISSSystem:
    """실제 PDF 데이터를 활용한 완전한 FAISS HNSW 시스템 (HNSW 명시적 구현 + Gemini 임베딩 지원)"""
    
    def __init__(self, embedding_type: str = "huggingface"):
        """
        초기화
        Args:
            embedding_type: "huggingface" 또는 "gemini"
        """
        self.pdf_path = Path("DocumentsLoader/data/기술적차트분석이론및방법.pdf")
        self.index_dir = Path("DocumentsLoader/faiss_index_complete")
        self.index_dir.mkdir(exist_ok=True)
        
        # 텍스트 분할 설정 (실제 PDF에 최적화)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
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
        self.pdf_processor = ImprovedPDFProcessor()
        
        # HNSW 설정
        self.hnsw_config = {
            'M': 16,  # 각 노드의 최대 연결 수
            'efConstruction': 200,  # 구축 시 탐색할 이웃 수
            'efSearch': 50,  # 검색 시 탐색할 이웃 수
            'metric': faiss.METRIC_INNER_PRODUCT  # 코사인 유사도
        }
        
        # 기술적 분석 검색 최적화 키워드
        self.search_keywords = {
            'RSI': ['RSI', '상대강도지수', '상대강도', '과매수', '과매도', '70', '30'],
            'MACD': ['MACD', '이동평균수렴확산', '이동평균수렴', '골든크로스', '데드크로스'],
            '볼린저밴드': ['볼린저밴드', '볼린저', '변동성', '밴드', '표준편차'],
            '이동평균선': ['이동평균선', '이평선', '이동평균', '이평', '추세'],
            '스토캐스틱': ['스토캐스틱', '오실레이터', '%K', '%D'],
            '일목균형표': ['일목균형표', '일목균형', '구름대', '시간론', '가격론'],
            '피보나치': ['피보나치', '되돌림', '23.6', '38.2', '61.8', '황금비율'],
            '엘리어트': ['엘리어트', '파동이론', '파동', '상승파', '하락파'],
            '지지저항': ['지지선', '저항선', '지지', '저항', '지지대', '저항대'],
            '거래량': ['거래량', '거래', '활성도', '매물대', '세력활동']
        }
    
    def load_pdf_documents(self) -> List[Document]:
        """PDF 문서 로드 및 LangChain Document 변환"""
        print("📄 PDF 문서 로드 중...")
        
        try:
            # PDF 존재 확인
            if not self.pdf_processor.check_pdf_exists():
                print("❌ PDF 파일을 찾을 수 없습니다.")
                return []
            
            # PDF 로드
            docs = self.pdf_processor.load_pdf_with_pypdf()
            if not docs:
                print("❌ PDF 로드에 실패했습니다.")
                return []
            
            # LangChain Document 형식으로 변환
            langchain_docs = []
            for i, doc in enumerate(docs):
                # 메타데이터 구성
                metadata = {
                    'source': str(self.pdf_path),
                    'page': i + 1,
                    'total_pages': len(docs),
                    'content_type': 'technical_analysis',
                    'language': 'ko',
                    'domain': 'stock_analysis',
                    'embedding_type': self.embedding_type,
                    'file_size': self.pdf_path.stat().st_size,
                    'processing_time': datetime.now().isoformat()
                }
                
                # 기존 메타데이터 병합
                if hasattr(doc, 'metadata') and doc.metadata:
                    metadata.update(doc.metadata)
                
                # LangChain Document 생성
                langchain_doc = Document(
                    page_content=doc.page_content,
                    metadata=metadata
                )
                langchain_docs.append(langchain_doc)
            
            print(f"   ✅ {len(langchain_docs)}개 문서 로드 완료")
            return langchain_docs
            
        except Exception as e:
            print(f"   ❌ PDF 로드 실패: {str(e)}")
            return []
    
    def split_and_process_documents(self, documents: List[Document]) -> List[Document]:
        """문서 분할 및 전처리"""
        print("✂️ 문서 분할 및 전처리 중...")
        
        try:
            # 문서 분할
            split_docs = self.text_splitter.split_documents(documents)
            
            # 청크별 메타데이터 추가
            for i, doc in enumerate(split_docs):
                doc.metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(doc.page_content),
                    'chunk_processing_time': datetime.now().isoformat(),
                    'has_technical_content': self._check_technical_content(doc.page_content),
                    'embedding_type': self.embedding_type
                })
            
            print(f"   ✅ {len(split_docs)}개 청크 생성 완료")
            print(f"   📊 평균 청크 크기: {sum(len(doc.page_content) for doc in split_docs) // len(split_docs)}자")
            
            # 기술적 분석 내용이 포함된 청크 수 계산
            technical_chunks = sum(1 for doc in split_docs if doc.metadata.get('has_technical_content', False))
            print(f"   🔍 기술적 분석 내용 포함 청크: {technical_chunks}개")
            
            return split_docs
            
        except Exception as e:
            print(f"   ❌ 문서 분할 실패: {str(e)}")
            return []
    
    def _check_technical_content(self, content: str) -> bool:
        """기술적 분석 내용 포함 여부 확인"""
        technical_terms = [
            'RSI', 'MACD', '볼린저', '이동평균', '스토캐스틱', '일목균형표',
            '피보나치', '엘리어트', '지지선', '저항선', '거래량', '추세',
            '과매수', '과매도', '골든크로스', '데드크로스', '다이버전스'
        ]
        
        content_lower = content.lower()
        return any(term.lower() in content_lower for term in technical_terms)
    
    def create_faiss_index(self, documents: List[Document]) -> bool:
        """FAISS HNSW 인덱스 생성 (명시적 HNSW 구현)"""
        print("🔍 FAISS HNSW 인덱스 생성 중...")
        print(f"   📊 임베딩 모델: {self.embedding_type}")
        print(f"   🔧 HNSW 설정: M={self.hnsw_config['M']}, efConstruction={self.hnsw_config['efConstruction']}")
        
        try:
            # 임베딩 생성
            print("   🔄 임베딩 생성 중...")
            embeddings_list = []
            for doc in documents:
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
                docstore=self._create_docstore(documents),
                index_to_docstore_id={i: i for i in range(len(documents))}
            )
            
            # 인덱스 저장
            self.faiss_index.save_local(str(self.index_dir))
            
            print(f"   ✅ FAISS HNSW 인덱스 생성 완료")
            print(f"   📁 저장 위치: {self.index_dir}")
            print(f"   📊 총 벡터 수: {index.ntotal}")
            print(f"   🔧 HNSW 노드 수: {index.hnsw.levels.size()}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ FAISS 인덱스 생성 실패: {str(e)}")
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
    
    def load_existing_index(self) -> bool:
        """기존 인덱스 로드"""
        try:
            if (self.index_dir / "index.faiss").exists():
                print("📂 기존 FAISS HNSW 인덱스 로드 중...")
                self.faiss_index = FAISS.load_local(
                    str(self.index_dir),
                    self.embeddings
                )
                print("   ✅ 기존 인덱스 로드 완료")
                
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
    
    def semantic_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """의미론적 검색 (HNSW 최적화)"""
        if not self.faiss_index:
            print("❌ FAISS 인덱스가 로드되지 않았습니다.")
            return []
        
        try:
            print(f"🔍 HNSW 의미론적 검색: '{query}'")
            
            # HNSW 검색 파라미터 설정
            if hasattr(self.faiss_index.index, 'hnsw'):
                self.faiss_index.index.hnsw.efSearch = self.hnsw_config['efSearch']
            
            # 유사도 검색 수행
            docs_and_scores = self.faiss_index.similarity_search_with_score(
                query, k=k
            )
            
            print(f"   ✅ {len(docs_and_scores)}개 결과 발견")
            
            # 결과 출력
            for i, (doc, score) in enumerate(docs_and_scores, 1):
                print(f"   📄 결과 {i}:")
                print(f"      - 유사도 점수: {score:.4f}")
                print(f"      - 페이지: {doc.metadata.get('page', 'N/A')}")
                print(f"      - 청크 크기: {doc.metadata.get('chunk_size', 'N/A')}자")
                print(f"      - 임베딩 타입: {doc.metadata.get('embedding_type', 'N/A')}")
                print(f"      - 기술적 내용: {'✅' if doc.metadata.get('has_technical_content', False) else '❌'}")
                print(f"      - 내용 미리보기: {doc.page_content[:100]}...")
                print()
            
            return docs_and_scores
            
        except Exception as e:
            print(f"   ❌ 검색 실패: {str(e)}")
            return []
    
    def technical_search(self, indicator: str, k: int = 5) -> List[Tuple[Document, float]]:
        """기술적 분석 특화 검색 (HNSW 최적화)"""
        if indicator not in self.search_keywords:
            print(f"❌ 지원하지 않는 지표입니다: {indicator}")
            return []
        
        print(f"📈 HNSW 기술적 분석 검색: {indicator}")
        
        # 관련 키워드들로 검색
        keywords = self.search_keywords[indicator]
        query = " ".join(keywords)
        
        print(f"   🔍 검색 키워드: {', '.join(keywords)}")
        
        # 의미론적 검색 수행
        results = self.semantic_search(query, k=k*2)  # 더 많은 결과에서 필터링
        
        # 기술적 내용이 포함된 결과만 필터링
        filtered_results = []
        for doc, score in results:
            if doc.metadata.get('has_technical_content', False):
                # 관련 키워드 매칭 점수 추가
                content_lower = doc.page_content.lower()
                keyword_matches = sum(1 for keyword in keywords if keyword.lower() in content_lower)
                enhanced_score = score + (keyword_matches * 0.1)
                filtered_results.append((doc, enhanced_score))
        
        # 점수로 재정렬
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   ✅ 기술적 분석 결과: {len(filtered_results)}개")
        return filtered_results[:k]
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """인덱스 통계 정보"""
        if not self.faiss_index:
            return {}
        
        try:
            # 인덱스 정보 수집
            stats = {
                'embedding_type': self.embedding_type,
                'total_vectors': len(self.faiss_index.docstore._dict),
                'index_type': 'FAISS HNSW',
                'hnsw_config': self.hnsw_config,
                'index_path': str(self.index_dir),
                'created_time': datetime.now().isoformat()
            }
            
            # HNSW 특정 정보
            if hasattr(self.faiss_index.index, 'hnsw'):
                stats.update({
                    'hnsw_nodes': self.faiss_index.index.hnsw.levels.size(),
                    'hnsw_max_level': self.faiss_index.index.hnsw.max_level,
                    'hnsw_ef_search': self.faiss_index.index.hnsw.efSearch,
                    'hnsw_ef_construction': self.faiss_index.index.hnsw.efConstruction
                })
            
            # 문서 통계
            docs = list(self.faiss_index.docstore._dict.values())
            if docs:
                technical_docs = sum(1 for doc in docs if doc.metadata.get('has_technical_content', False))
                stats.update({
                    'total_documents': len(docs),
                    'technical_documents': technical_docs,
                    'avg_document_length': sum(len(doc.page_content) for doc in docs) // len(docs),
                    'total_characters': sum(len(doc.page_content) for doc in docs),
                    'pages_covered': len(set(doc.metadata.get('page', 0) for doc in docs))
                })
            
            return stats
            
        except Exception as e:
            print(f"❌ 통계 수집 실패: {str(e)}")
            return {}
    
    def save_search_results(self, results: List[Tuple[Document, float]], query: str, search_type: str = "semantic"):
        """검색 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_results_{search_type}_{timestamp}.json"
        
        try:
            # 결과를 JSON 형식으로 변환
            search_data = {
                'query': query,
                'search_type': search_type,
                'embedding_type': self.embedding_type,
                'hnsw_config': self.hnsw_config,
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
                    'embedding_type': doc.metadata.get('embedding_type', 'N/A'),
                    'has_technical_content': doc.metadata.get('has_technical_content', False),
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                search_data['results'].append(result_item)
            
            # 파일 저장
            output_path = Path("DocumentsLoader/search_results") / filename
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(search_data, f, ensure_ascii=False, indent=2)
            
            print(f"   ✅ 검색 결과 저장: {output_path}")
            
        except Exception as e:
            print(f"   ❌ 결과 저장 실패: {str(e)}")
    
    def build_index(self) -> bool:
        """전체 인덱스 구축 프로세스"""
        print("🚀 완전한 FAISS HNSW 인덱스 구축 시작")
        print(f"📊 임베딩 모델: {self.embedding_type}")
        print("=" * 80)
        
        # 1. 기존 인덱스 확인
        if self.load_existing_index():
            print("✅ 기존 인덱스 사용 가능")
            return True
        
        # 2. PDF 문서 로드
        documents = self.load_pdf_documents()
        if not documents:
            return False
        
        # 3. 문서 분할 및 전처리
        split_docs = self.split_and_process_documents(documents)
        if not split_docs:
            return False
        
        # 4. FAISS 인덱스 생성
        success = self.create_faiss_index(split_docs)
        
        if success:
            # 5. 통계 정보 출력
            stats = self.get_index_statistics()
            print(f"\n📊 인덱스 통계:")
            for key, value in stats.items():
                print(f"   • {key}: {value}")
        
        return success
    
    def run_demo(self):
        """데모 검색 실행"""
        print("\n🧪 HNSW 데모 검색 실행")
        print("=" * 50)
        
        # 1. RSI 검색
        print("\n1. RSI 기술적 분석 검색:")
        rsi_results = self.technical_search("RSI")
        if rsi_results:
            self.save_search_results(rsi_results, "RSI", "technical")
        
        # 2. MACD 검색
        print("\n2. MACD 기술적 분석 검색:")
        macd_results = self.technical_search("MACD")
        if macd_results:
            self.save_search_results(macd_results, "MACD", "technical")
        
        # 3. 볼린저밴드 검색
        print("\n3. 볼린저밴드 기술적 분석 검색:")
        bb_results = self.technical_search("볼린저밴드")
        if bb_results:
            self.save_search_results(bb_results, "볼린저밴드", "technical")
        
        # 4. 이동평균선 검색
        print("\n4. 이동평균선 기술적 분석 검색:")
        ma_results = self.technical_search("이동평균선")
        if ma_results:
            self.save_search_results(ma_results, "이동평균선", "technical")
        
        # 5. 스토캐스틱 검색
        print("\n5. 스토캐스틱 기술적 분석 검색:")
        stoch_results = self.technical_search("스토캐스틱")
        if stoch_results:
            self.save_search_results(stoch_results, "스토캐스틱", "technical")
        
        # 6. 의미론적 검색 예시
        print("\n6. 의미론적 검색 예시:")
        semantic_results = self.semantic_search("기술적 분석 지표의 활용 방법")
        if semantic_results:
            self.save_search_results(semantic_results, "기술적 분석 지표의 활용 방법", "semantic")
    
    def interactive_search(self):
        """대화형 검색 인터페이스"""
        print("\n🔍 HNSW 대화형 검색 인터페이스")
        print("=" * 50)
        print("사용 가능한 명령어:")
        print("  - 'search <검색어>': 의미론적 검색")
        print("  - 'technical <지표명>': 기술적 분석 검색")
        print("  - 'stats': 인덱스 통계")
        print("  - 'quit': 종료")
        print(f"\n현재 임베딩 모델: {self.embedding_type}")
        print("\n지원하는 지표: RSI, MACD, 볼린저밴드, 이동평균선, 스토캐스틱, 일목균형표, 피보나치, 엘리어트, 지지저항, 거래량")
        print()
        
        while True:
            try:
                command = input("검색 명령어 입력: ").strip()
                
                if command.lower() == 'quit':
                    break
                elif command.lower() == 'stats':
                    stats = self.get_index_statistics()
                    print(f"\n📊 인덱스 통계:")
                    for key, value in stats.items():
                        print(f"   • {key}: {value}")
                
                elif command.startswith('search '):
                    query = command[7:].strip()
                    results = self.semantic_search(query)
                    if results:
                        self.save_search_results(results, query, "semantic")
                
                elif command.startswith('technical '):
                    indicator = command[10:].strip()
                    if indicator in self.search_keywords:
                        results = self.technical_search(indicator)
                        if results:
                            self.save_search_results(results, indicator, "technical")
                    else:
                        print(f"❌ 지원하지 않는 지표입니다: {indicator}")
                        print(f"   지원하는 지표: {', '.join(self.search_keywords.keys())}")
                
                else:
                    print("❌ 잘못된 명령어입니다. 다시 시도해주세요.")
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 검색 인터페이스를 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")

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
    
    print(f"🔧 사용할 임베딩 모델: {embedding_type}")
    
    # 완전한 FAISS 시스템 초기화
    faiss_system = CompleteFAISSSystem(embedding_type=embedding_type)
    
    # 인덱스 구축
    if faiss_system.build_index():
        print("\n🎉 완전한 FAISS HNSW 인덱스 구축 완료!")
        
        # 데모 검색 실행
        faiss_system.run_demo()
        
        # 대화형 검색 인터페이스
        print("\n🔍 대화형 검색 인터페이스 시작:")
        faiss_system.interactive_search()
        
    else:
        print("❌ 인덱스 구축에 실패했습니다.")

if __name__ == "__main__":
    main() 