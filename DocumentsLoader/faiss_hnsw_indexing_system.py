# LangChain을 활용한 FAISS HNSW 인덱싱 시스템
# 기술적 분석 PDF 내용을 벡터화하여 효율적인 검색 시스템 구축

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

# LangChain 관련 임포트
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# 기존 PDF 처리 시스템 임포트
from process_technical_analysis_pdf_improved import ImprovedPDFProcessor

print("🔧 LangChain FAISS HNSW 인덱싱 시스템 초기화")
print("=" * 60)

class FAISSHNSWIndexingSystem:
    """LangChain을 활용한 FAISS HNSW 인덱싱 시스템"""
    
    def __init__(self):
        self.pdf_path = Path("DocumentsLoader/data/기술적차트분석이론및방법.pdf")
        self.index_dir = Path("DocumentsLoader/faiss_index")
        self.index_dir.mkdir(exist_ok=True)
        
        # 텍스트 분할 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # 임베딩 모델 설정 (한국어 최적화)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # FAISS 인덱스 설정 (HNSW 방식)
        self.faiss_index = None
        self.documents = []
        
        # 기술적 분석 키워드 (검색 최적화)
        self.technical_keywords = {
            'indicators': [
                'RSI', '상대강도지수', 'MACD', '이동평균수렴확산',
                '볼린저밴드', 'Bollinger', '스토캐스틱', 'Stochastic',
                '이동평균선', '이평선', '이동평균', '이평',
                '일목균형표', '일목균형', '균형표',
                '피보나치', 'Fibonacci', '피보나치되돌림', '피보나치확장',
                '엘리어트', 'Elliott', '엘리어트파동', '파동이론'
            ],
            'concepts': [
                '지지선', '저항선', '추세선', '과매수', '과매도', '다이버전스',
                '골든크로스', '데드크로스', '거래량', '매물대',
                '지지', '저항', '추세', '추세대',
                '크로스', '크로스오버', '브레이크아웃', '브레이크', '돌파',
                '풀백', '되돌림', '조정', '반등'
            ]
        }
    
    def load_and_process_pdf(self) -> List[Document]:
        """PDF 로드 및 전처리"""
        print("📄 PDF 로드 및 전처리 중...")
        
        try:
            # 기존 PDF 처리 시스템 활용
            pdf_processor = ImprovedPDFProcessor()
            
            if not pdf_processor.check_pdf_exists():
                print("❌ PDF 파일을 찾을 수 없습니다.")
                return []
            
            # PDF 로드
            docs = pdf_processor.load_pdf_with_pypdf()
            if not docs:
                print("❌ PDF 로드에 실패했습니다.")
                return []
            
            # LangChain Document 형식으로 변환
            langchain_docs = []
            for i, doc in enumerate(docs):
                # 메타데이터 추가
                metadata = {
                    'source': str(self.pdf_path),
                    'page': i + 1,
                    'total_pages': len(docs),
                    'content_type': 'technical_analysis',
                    'language': 'ko',
                    'domain': 'stock_analysis'
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
            print(f"   ❌ PDF 처리 실패: {str(e)}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """문서를 청크로 분할"""
        print("✂️ 문서 청크 분할 중...")
        
        try:
            split_docs = self.text_splitter.split_documents(documents)
            
            # 청크별 메타데이터 추가
            for i, doc in enumerate(split_docs):
                doc.metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(doc.page_content),
                    'processing_time': datetime.now().isoformat()
                })
            
            print(f"   ✅ {len(split_docs)}개 청크 생성 완료")
            print(f"   📊 평균 청크 크기: {sum(len(doc.page_content) for doc in split_docs) // len(split_docs)}자")
            
            return split_docs
            
        except Exception as e:
            print(f"   ❌ 문서 분할 실패: {str(e)}")
            return []
    
    def create_faiss_index(self, documents: List[Document]) -> bool:
        """FAISS HNSW 인덱스 생성"""
        print("🔍 FAISS HNSW 인덱스 생성 중...")
        
        try:
            # FAISS 벡터 스토어 생성 (HNSW 방식)
            self.faiss_index = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name="technical_analysis_hnsw"
            )
            
            # 인덱스 저장
            self.faiss_index.save_local(str(self.index_dir))
            
            print(f"   ✅ FAISS 인덱스 생성 완료")
            print(f"   📁 저장 위치: {self.index_dir}")
            print(f"   📊 총 벡터 수: {len(documents)}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ FAISS 인덱스 생성 실패: {str(e)}")
            return False
    
    def load_existing_index(self) -> bool:
        """기존 인덱스 로드"""
        try:
            if (self.index_dir / "index.faiss").exists():
                print("📂 기존 FAISS 인덱스 로드 중...")
                self.faiss_index = FAISS.load_local(
                    str(self.index_dir),
                    self.embeddings,
                    index_name="technical_analysis_hnsw"
                )
                print("   ✅ 기존 인덱스 로드 완료")
                return True
            else:
                print("   ⚠️ 기존 인덱스가 없습니다.")
                return False
        except Exception as e:
            print(f"   ❌ 인덱스 로드 실패: {str(e)}")
            return False
    
    def semantic_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """의미론적 검색 수행"""
        if not self.faiss_index:
            print("❌ FAISS 인덱스가 로드되지 않았습니다.")
            return []
        
        try:
            print(f"🔍 의미론적 검색: '{query}'")
            
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
                print(f"      - 내용 미리보기: {doc.page_content[:100]}...")
                print()
            
            return docs_and_scores
            
        except Exception as e:
            print(f"   ❌ 검색 실패: {str(e)}")
            return []
    
    def keyword_search(self, keywords: List[str], k: int = 5) -> List[Tuple[Document, float]]:
        """키워드 기반 검색"""
        if not self.faiss_index:
            print("❌ FAISS 인덱스가 로드되지 않았습니다.")
            return []
        
        try:
            print(f"🔍 키워드 검색: {', '.join(keywords)}")
            
            # 키워드를 결합한 쿼리 생성
            query = " ".join(keywords)
            
            # 의미론적 검색 수행
            results = self.semantic_search(query, k=k)
            
            # 키워드 매칭 필터링
            filtered_results = []
            for doc, score in results:
                content_lower = doc.page_content.lower()
                keyword_matches = sum(1 for keyword in keywords if keyword.lower() in content_lower)
                
                if keyword_matches > 0:
                    # 키워드 매칭 점수 추가
                    enhanced_score = score + (keyword_matches * 0.1)
                    filtered_results.append((doc, enhanced_score))
            
            # 점수로 재정렬
            filtered_results.sort(key=lambda x: x[1], reverse=True)
            
            print(f"   ✅ 키워드 매칭 결과: {len(filtered_results)}개")
            return filtered_results[:k]
            
        except Exception as e:
            print(f"   ❌ 키워드 검색 실패: {str(e)}")
            return []
    
    def technical_analysis_search(self, indicator: str, concept: str = None) -> List[Tuple[Document, float]]:
        """기술적 분석 특화 검색"""
        print(f"📈 기술적 분석 검색: {indicator}")
        if concept:
            print(f"   관련 개념: {concept}")
        
        # 검색 쿼리 구성
        search_terms = [indicator]
        if concept:
            search_terms.append(concept)
        
        # 한국어 설명 추가
        korean_explanations = {
            'RSI': '상대강도지수',
            'MACD': '이동평균수렴확산',
            '볼린저밴드': '볼린저밴드',
            '이동평균선': '이동평균선',
            '스토캐스틱': '스토캐스틱',
            '일목균형표': '일목균형표',
            '피보나치': '피보나치되돌림',
            '엘리어트': '엘리어트파동'
        }
        
        if indicator in korean_explanations:
            search_terms.append(korean_explanations[indicator])
        
        return self.keyword_search(search_terms, k=10)
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """인덱스 통계 정보"""
        if not self.faiss_index:
            return {}
        
        try:
            # 인덱스 정보 수집
            stats = {
                'total_vectors': len(self.faiss_index.docstore._dict),
                'index_type': 'FAISS HNSW',
                'embedding_model': 'paraphrase-multilingual-MiniLM-L12-v2',
                'index_path': str(self.index_dir),
                'created_time': datetime.now().isoformat()
            }
            
            # 문서 통계
            docs = list(self.faiss_index.docstore._dict.values())
            if docs:
                stats.update({
                    'total_documents': len(docs),
                    'avg_document_length': sum(len(doc.page_content) for doc in docs) // len(docs),
                    'total_characters': sum(len(doc.page_content) for doc in docs),
                    'pages_covered': len(set(doc.metadata.get('page', 0) for doc in docs))
                })
            
            return stats
            
        except Exception as e:
            print(f"❌ 통계 수집 실패: {str(e)}")
            return {}
    
    def save_search_results(self, results: List[Tuple[Document, float]], query: str, filename: str = None):
        """검색 결과 저장"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"search_results_{timestamp}.json"
        
        try:
            # 결과를 JSON 형식으로 변환
            search_data = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'total_results': len(results),
                'results': []
            }
            
            for doc, score in results:
                result_item = {
                    'score': float(score),
                    'page': doc.metadata.get('page', 'N/A'),
                    'chunk_id': doc.metadata.get('chunk_id', 'N/A'),
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
        print("🚀 FAISS HNSW 인덱스 구축 시작")
        print("=" * 60)
        
        # 1. 기존 인덱스 확인
        if self.load_existing_index():
            print("✅ 기존 인덱스 사용 가능")
            return True
        
        # 2. PDF 로드 및 처리
        documents = self.load_and_process_pdf()
        if not documents:
            return False
        
        # 3. 문서 청크 분할
        split_docs = self.split_documents(documents)
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
    
    def interactive_search(self):
        """대화형 검색 인터페이스"""
        print("\n🔍 대화형 검색 인터페이스")
        print("=" * 40)
        print("사용 가능한 명령어:")
        print("  - 'search <검색어>': 의미론적 검색")
        print("  - 'keyword <키워드1,키워드2>': 키워드 검색")
        print("  - 'technical <지표명>': 기술적 분석 검색")
        print("  - 'stats': 인덱스 통계")
        print("  - 'quit': 종료")
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
                        self.save_search_results(results, query)
                
                elif command.startswith('keyword '):
                    keywords_str = command[8:].strip()
                    keywords = [k.strip() for k in keywords_str.split(',')]
                    results = self.keyword_search(keywords)
                    if results:
                        self.save_search_results(results, f"keywords: {keywords_str}")
                
                elif command.startswith('technical '):
                    indicator = command[10:].strip()
                    results = self.technical_analysis_search(indicator)
                    if results:
                        self.save_search_results(results, f"technical: {indicator}")
                
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
    # FAISS 인덱싱 시스템 초기화
    indexing_system = FAISSHNSWIndexingSystem()
    
    # 인덱스 구축
    if indexing_system.build_index():
        print("\n🎉 FAISS HNSW 인덱스 구축 완료!")
        
        # 샘플 검색 테스트
        print("\n🧪 샘플 검색 테스트")
        
        # 1. 의미론적 검색 테스트
        print("\n1. RSI 의미론적 검색:")
        indexing_system.semantic_search("RSI 상대강도지수 분석 방법")
        
        # 2. 키워드 검색 테스트
        print("\n2. MACD 키워드 검색:")
        indexing_system.keyword_search(["MACD", "이동평균수렴확산", "골든크로스"])
        
        # 3. 기술적 분석 검색 테스트
        print("\n3. 볼린저밴드 기술적 분석 검색:")
        indexing_system.technical_analysis_search("볼린저밴드", "과매수")
        
        # 4. 대화형 검색 인터페이스
        print("\n4. 대화형 검색 인터페이스 시작:")
        indexing_system.interactive_search()
        
    else:
        print("❌ 인덱스 구축에 실패했습니다.")

if __name__ == "__main__":
    main() 