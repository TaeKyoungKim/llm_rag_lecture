"""
FAISS HNSW 인덱스 구축기
PDF 문서를 로드하고 임베딩하여 FAISS HNSW 인덱스를 구축하고 저장하는 시스템
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# PDF 처리 라이브러리
from PyPDF2 import PdfReader

from dotenv import load_dotenv
load_dotenv()

class FAISSIndexBuilder:
    """
    FAISS HNSW 인덱스 구축 클래스
    PDF 문서를 처리하여 FAISS 인덱스를 생성하고 저장
    """
    
    def __init__(self, embedding_type: str = "huggingface"):
        """
        인덱스 구축기 초기화
        Args:
            embedding_type: "huggingface" 또는 "gemini"
        """
        print("🔧 FAISS 인덱스 구축기 초기화")
        
        # 파일 경로 설정
        self.pdf_path = Path("DocumentsLoader/data/기술적차트분석이론및방법.pdf")
        self.index_dir = Path("DocumentsLoader/educational_faiss_index")
        self.index_dir.mkdir(exist_ok=True)
        
        # 임베딩 타입 설정
        self.embedding_type = embedding_type
        
        # HNSW 설정
        self.hnsw_config = {
            'M': 16,  # 각 노드의 최대 연결 수
            'efConstruction': 100,  # 구축 시 탐색할 이웃 수
            'efSearch': 50,  # 검색 시 탐색할 이웃 수
            'metric': faiss.METRIC_INNER_PRODUCT  # 코사인 유사도
        }
        
        # 임베딩 모델 초기화
        self._initialize_embeddings()
        
        # 텍스트 분할기 초기화
        self._initialize_text_splitter()
        
        print("   ✅ 인덱스 구축기 초기화 완료")
    
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
    
    def _initialize_text_splitter(self):
        """텍스트 분할기 초기화"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        print("   ✅ 텍스트 분할기 초기화 완료")
    
    def load_documents(self) -> List[Document]:
        """PDF 문서 로드"""
        print("\n📄 PDF 문서 로드")
        print("-" * 40)
        
        try:
            # PDF 파일 존재 확인
            if not self.pdf_path.exists():
                print(f"   ❌ PDF 파일을 찾을 수 없습니다: {self.pdf_path}")
                return []
            
            print(f"   📁 PDF 파일 경로: {self.pdf_path}")
            
            # PDF 읽기
            reader = PdfReader(str(self.pdf_path))
            total_pages = len(reader.pages)
            print(f"   📊 총 페이지 수: {total_pages}")
            
            # 각 페이지를 Document로 변환
            documents = []
            for page_num, page in enumerate(reader.pages):
                # 페이지 텍스트 추출
                text = page.extract_text()
                
                # 메타데이터 구성
                metadata = {
                    'source': str(self.pdf_path),
                    'page': page_num + 1,
                    'total_pages': total_pages,
                    'content_type': 'technical_analysis',
                    'language': 'ko',
                    'processing_time': datetime.now().isoformat()
                }
                
                # LangChain Document 생성
                doc = Document(
                    page_content=text,
                    metadata=metadata
                )
                documents.append(doc)
                
                print(f"   📄 페이지 {page_num + 1} 로드 완료 ({len(text)}자)")
            
            print(f"   ✅ 총 {len(documents)}개 문서 로드 완료")
            return documents
            
        except Exception as e:
            print(f"   ❌ 문서 로드 실패: {str(e)}")
            return []
    
    def tokenize_and_chunk(self, documents: List[Document]) -> List[Document]:
        """토크나이징 및 청킹"""
        print("\n✂️ 토크나이징 및 청킹")
        print("-" * 40)
        
        try:
            print("   🔄 문서 분할 중...")
            
            # 문서 분할
            split_docs = self.text_splitter.split_documents(documents)
            
            print(f"   📊 원본 문서 수: {len(documents)}")
            print(f"   📊 분할된 청크 수: {len(split_docs)}")
            
            # 청크별 메타데이터 추가
            for i, doc in enumerate(split_docs):
                # 기존 메타데이터 유지
                original_metadata = doc.metadata.copy()
                
                # 청크 관련 메타데이터 추가
                doc.metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(doc.page_content),
                    'chunk_processing_time': datetime.now().isoformat(),
                    'embedding_type': self.embedding_type,
                    'has_technical_content': self._check_technical_content(doc.page_content)
                })
                
                print(f"   📄 청크 {i+1}: {len(doc.page_content)}자 "
                      f"({'✅' if doc.metadata['has_technical_content'] else '❌'} 기술적 내용)")
            
            # 통계 정보
            avg_chunk_size = sum(len(doc.page_content) for doc in split_docs) // len(split_docs)
            technical_chunks = sum(1 for doc in split_docs if doc.metadata['has_technical_content'])
            
            print(f"   📈 평균 청크 크기: {avg_chunk_size}자")
            print(f"   🔍 기술적 내용 포함 청크: {technical_chunks}개")
            
            print("   ✅ 토크나이징 및 청킹 완료")
            return split_docs
            
        except Exception as e:
            print(f"   ❌ 토크나이징 및 청킹 실패: {str(e)}")
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
    
    def create_embeddings(self, documents: List[Document]) -> List[np.ndarray]:
        """임베딩 생성"""
        print("\n🔢 임베딩 생성")
        print("-" * 40)
        
        try:
            print(f"   🔄 {self.embedding_type} 임베딩 모델로 벡터 생성 중...")
            
            embeddings_list = []
            failed_count = 0
            max_retries = 3
            
            for i, doc in enumerate(documents):
                # 재시도 로직
                for retry in range(max_retries):
                    try:
                        # 임베딩 생성
                        embedding = self.embeddings.embed_query(doc.page_content)
                        
                        # 임베딩 유효성 검사
                        if embedding and len(embedding) > 0:
                            embeddings_list.append(embedding)
                            break
                        else:
                            raise Exception("빈 임베딩 생성됨")
                            
                    except Exception as e:
                        if retry < max_retries - 1:
                            print(f"   ⚠️ 문서 {i+1} 임베딩 실패 (재시도 {retry+1}/{max_retries}): {str(e)}")
                            import time
                            time.sleep(2)
                        else:
                            print(f"   ❌ 문서 {i+1} 임베딩 최종 실패: {str(e)}")
                            failed_count += 1
                            # 실패한 경우 기본 벡터 생성 (0으로 채움)
                            if embeddings_list:
                                default_dim = len(embeddings_list[0])
                                embeddings_list.append([0.0] * default_dim)
                            else:
                                # 첫 번째 임베딩이 실패한 경우 기본 차원 사용
                                embeddings_list.append([0.0] * 384)
                
                # 진행률 표시
                if (i + 1) % 10 == 0 or i == len(documents) - 1:
                    success_rate = ((i + 1 - failed_count) / (i + 1)) * 100
                    print(f"   📊 진행률: {i + 1}/{len(documents)} ({((i + 1) / len(documents) * 100):.1f}%) - 성공률: {success_rate:.1f}%")
            
            # 임베딩 차원 확인
            if embeddings_list:
                dimension = len(embeddings_list[0])
                print(f"   📏 임베딩 차원: {dimension}")
                print(f"   📊 총 임베딩 수: {len(embeddings_list)}")
                print(f"   ⚠️ 실패한 임베딩 수: {failed_count}")
                
                if failed_count > 0:
                    print(f"   💡 {failed_count}개의 임베딩이 실패하여 기본값(0)으로 대체되었습니다.")
                
                print("   ✅ 임베딩 생성 완료")
                return embeddings_list
            else:
                raise Exception("모든 임베딩 생성에 실패했습니다.")
            
        except Exception as e:
            print(f"   ❌ 임베딩 생성 실패: {str(e)}")
            
            # 오류 유형별 안내
            error_msg = str(e).lower()
            if "timeout" in error_msg:
                print("   💡 타임아웃 오류입니다. 네트워크 연결을 확인하거나 더 작은 배치로 시도해보세요.")
            elif "quota" in error_msg:
                print("   💡 API 할당량 초과입니다. 잠시 후 다시 시도하거나 다른 API 키를 사용하세요.")
            elif "invalid_grant" in error_msg or "bad request" in error_msg:
                print("   💡 API 인증 오류입니다. API 키를 확인해주세요.")
            
            return []
    
    def create_hnsw_index(self, documents: List[Document], embeddings: List[np.ndarray]) -> bool:
        """HNSW 인덱스 생성 및 저장"""
        print("\n🔍 HNSW 인덱스 생성")
        print("-" * 40)
        
        try:
            if not embeddings:
                print("   ❌ 임베딩이 없습니다.")
                return False
            
            # 임베딩을 numpy 배열로 변환
            embeddings_array = np.array(embeddings, dtype=np.float32)
            dimension = embeddings_array.shape[1]
            
            print(f"   📊 임베딩 배열 형태: {embeddings_array.shape}")
            print(f"   🔧 HNSW 설정: M={self.hnsw_config['M']}, "
                  f"efConstruction={self.hnsw_config['efConstruction']}")
            
            # HNSW 인덱스 생성
            print("   🔨 HNSW 인덱스 생성 중...")
            index = faiss.IndexHNSWFlat(dimension, self.hnsw_config['M'])
            
            # HNSW 파라미터 설정
            index.hnsw.efConstruction = self.hnsw_config['efConstruction']
            index.hnsw.efSearch = self.hnsw_config['efSearch']
            index.metric_type = self.hnsw_config['metric']
            
            # 벡터를 인덱스에 추가
            print("   📥 벡터를 인덱스에 추가 중...")
            index.add(embeddings_array)
            
            # LangChain FAISS 래퍼 생성
            print("   🔗 LangChain FAISS 래퍼 생성 중...")
            faiss_index = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=self._create_docstore(documents),
                index_to_docstore_id={i: i for i in range(len(documents))}
            )
            
            # 인덱스 저장
            print("   💾 인덱스 저장 중...")
            faiss_index.save_local(str(self.index_dir))
            
            # 저장된 인덱스 테스트 로드 (안전성 확인)
            print("   🔍 저장된 인덱스 테스트 로드 중...")
            test_load = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("   ✅ 인덱스 저장 및 테스트 로드 완료")
            
            # HNSW 정보 출력
            print(f"   📊 총 벡터 수: {index.ntotal}")
            print(f"   🔧 HNSW 노드 수: {index.hnsw.levels.size()}")
            print(f"   🔧 HNSW 최대 레벨: {index.hnsw.max_level}")
            print(f"   📁 저장 위치: {self.index_dir}")
            
            print("   ✅ HNSW 인덱스 생성 및 저장 완료")
            return True
            
        except Exception as e:
            print(f"   ❌ HNSW 인덱스 생성 실패: {str(e)}")
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
    
    def build_index(self) -> bool:
        """전체 인덱스 구축 프로세스"""
        print("\n🚀 FAISS HNSW 인덱스 구축 시작")
        print("=" * 80)
        
        # 1단계: 문서 로드
        documents = self.load_documents()
        if not documents:
            print("❌ 문서 로드 실패")
            return False
        
        # 2단계: 토크나이징 및 청킹
        chunked_docs = self.tokenize_and_chunk(documents)
        if not chunked_docs:
            print("❌ 토크나이징 및 청킹 실패")
            return False
        
        # 3단계: 임베딩 생성
        embeddings = self.create_embeddings(chunked_docs)
        if not embeddings:
            print("❌ 임베딩 생성 실패")
            return False
        
        # 4단계: HNSW 인덱스 생성 및 저장
        success = self.create_hnsw_index(chunked_docs, embeddings)
        
        if success:
            print("\n🎉 FAISS HNSW 인덱스 구축 완료!")
            self._print_build_statistics(chunked_docs, embeddings)
        
        return success
    
    def _print_build_statistics(self, documents: List[Document], embeddings: List[np.ndarray]):
        """구축 통계 정보 출력"""
        print("\n📊 구축 통계:")
        print("-" * 30)
        
        # 기본 정보
        print(f"   • 임베딩 모델: {self.embedding_type}")
        print(f"   • 인덱스 타입: FAISS HNSW")
        print(f"   • 저장 위치: {self.index_dir}")
        
        # 문서 정보
        if documents:
            technical_docs = sum(1 for doc in documents if doc.metadata.get('has_technical_content', False))
            print(f"   • 총 문서 수: {len(documents)}")
            print(f"   • 기술적 내용 문서: {technical_docs}")
            print(f"   • 평균 문서 길이: {sum(len(doc.page_content) for doc in documents) // len(documents)}자")
        
        # 임베딩 정보
        if embeddings:
            print(f"   • 임베딩 차원: {len(embeddings[0])}")
            print(f"   • 총 임베딩 수: {len(embeddings)}")

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
    
    print(f"🔧 FAISS 인덱스 구축기 - 임베딩 모델: {embedding_type}")
    
    # 인덱스 구축기 초기화
    builder = FAISSIndexBuilder(embedding_type=embedding_type)
    
    # 인덱스 구축
    if builder.build_index():
        print("\n🎉 인덱스 구축이 완료되었습니다!")
        print("💡 이제 'faiss_search_engine.py'를 사용하여 검색할 수 있습니다.")
    else:
        print("❌ 인덱스 구축에 실패했습니다.")

if __name__ == "__main__":
    main() 