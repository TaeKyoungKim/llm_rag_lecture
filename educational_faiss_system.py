"""
교육용 FAISS HNSW 시스템
단계별로 문서 로딩, 토크나이징, 청킹, 임베딩, HNSW 저장, 검색을 구현
"""

import os
import json
import re
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

print("🎓 교육용 FAISS HNSW 시스템 시작")
print("=" * 80)
from dotenv import load_dotenv
load_dotenv()

class EducationalFAISSSystem:
    """
    교육 목적의 FAISS HNSW 시스템
    각 단계를 명확히 구분하여 RAG 시스템의 전체 과정을 학습할 수 있도록 구현
    """
    
    def __init__(self, embedding_type: str = "huggingface"):
        """
        시스템 초기화
        Args:
            embedding_type: "huggingface" 또는 "gemini"
        """
        print("🔧 1단계: 시스템 초기화")
        
        # 파일 경로 설정
        self.pdf_path = Path("DocumentsLoader/data/기술적차트분석이론및방법.pdf")
        self.index_dir = Path("DocumentsLoader/educational_faiss_index")
        self.index_dir.mkdir(exist_ok=True)
        
        # 임베딩 타입 설정
        self.embedding_type = embedding_type
        
        # HNSW 설정 (교육용으로 간단하게 설정)
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
        
        # FAISS 인덱스 초기화
        self.faiss_index = None
        
        print("   ✅ 시스템 초기화 완료")
    
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
                    task_type="retrieval_query",  # 검색 최적화
                    title="Technical Analysis Document"  # 문서 제목
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
            chunk_size=800,  # 각 청크의 최대 크기
            chunk_overlap=150,  # 청크 간 겹치는 부분
            length_function=len,  # 길이 측정 함수
            separators=["\n\n", "\n", " ", ""]  # 분할 기준 구분자
        )
        print("   ✅ 텍스트 분할기 초기화 완료")
    
    def step1_load_documents(self) -> List[Document]:
        """
        2단계: PDF 문서 로드
        PDF 파일을 읽어서 LangChain Document 형식으로 변환
        """
        print("\n📄 2단계: PDF 문서 로드")
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
    
    def step2_tokenize_and_chunk(self, documents: List[Document]) -> List[Document]:
        """
        3단계: 토크나이징 및 청킹
        문서를 작은 청크로 분할하여 처리 효율성 향상
        """
        print("\n✂️ 3단계: 토크나이징 및 청킹")
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
    
    def step3_create_embeddings(self, documents: List[Document]) -> List[np.ndarray]:
        """
        4단계: 임베딩 생성
        각 문서 청크를 벡터로 변환
        """
        print("\n🔢 4단계: 임베딩 생성")
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
                            time.sleep(2)  # 잠시 대기
                        else:
                            print(f"   ❌ 문서 {i+1} 임베딩 최종 실패: {str(e)}")
                            failed_count += 1
                            # 실패한 경우 기본 벡터 생성 (0으로 채움)
                            if embeddings_list:
                                default_dim = len(embeddings_list[0])
                                embeddings_list.append([0.0] * default_dim)
                            else:
                                # 첫 번째 임베딩이 실패한 경우 기본 차원 사용
                                embeddings_list.append([0.0] * 384)  # 일반적인 임베딩 차원
                
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
    
    def step4_create_hnsw_index(self, documents: List[Document], embeddings: List[np.ndarray]) -> bool:
        """
        5단계: HNSW 인덱스 생성
        FAISS HNSW 인덱스를 생성하고 벡터를 저장
        """
        print("\n🔍 5단계: HNSW 인덱스 생성")
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
            self.faiss_index = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=self._create_docstore(documents),
                index_to_docstore_id={i: i for i in range(len(documents))}
            )
            
            # 인덱스 저장
            print("   💾 인덱스 저장 중...")
            self.faiss_index.save_local(str(self.index_dir))
            
            # HNSW 정보 출력
            print(f"   📊 총 벡터 수: {index.ntotal}")
            print(f"   🔧 HNSW 노드 수: {index.hnsw.levels.size()}")
            print(f"   🔧 HNSW 최대 레벨: {index.hnsw.max_level}")
            print(f"   📁 저장 위치: {self.index_dir}")
            
            print("   ✅ HNSW 인덱스 생성 완료")
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
    
    def step5_similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        6단계: 유사도 검색
        쿼리와 유사한 문서를 HNSW 인덱스에서 검색
        """
        print(f"\n🔍 6단계: 유사도 검색")
        print("-" * 40)
        
        if not self.faiss_index:
            print("   ❌ FAISS 인덱스가 로드되지 않았습니다.")
            return []
        
        try:
            print(f"   🔍 검색 쿼리: '{query}'")
            print(f"   📊 검색할 결과 수: {k}")
            
            # HNSW 검색 파라미터 설정
            if hasattr(self.faiss_index.index, 'hnsw'):
                self.faiss_index.index.hnsw.efSearch = self.hnsw_config['efSearch']
                print(f"   🔧 HNSW efSearch 설정: {self.hnsw_config['efSearch']}")
            
            # 유사도 검색 수행
            print("   🔄 검색 수행 중...")
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
                return True
            else:
                print("   ⚠️ 기존 인덱스가 없습니다.")
                return False
        except Exception as e:
            print(f"   ❌ 인덱스 로드 실패: {str(e)}")
            return False
    
    def build_complete_system(self) -> bool:
        """
        전체 시스템 구축
        모든 단계를 순차적으로 실행
        """
        print("\n🚀 전체 FAISS HNSW 시스템 구축 시작")
        print("=" * 80)
        
        # 기존 인덱스 확인
        if self.load_existing_index():
            print("✅ 기존 인덱스 사용 가능")
            return True
        
        # 1단계: 시스템 초기화 (이미 완료됨)
        print("✅ 1단계: 시스템 초기화 완료")
        
        # 2단계: 문서 로드
        documents = self.step1_load_documents()
        if not documents:
            print("❌ 문서 로드 실패")
            return False
        
        # 3단계: 토크나이징 및 청킹
        chunked_docs = self.step2_tokenize_and_chunk(documents)
        if not chunked_docs:
            print("❌ 토크나이징 및 청킹 실패")
            return False
        
        # 4단계: 임베딩 생성
        embeddings = self.step3_create_embeddings(chunked_docs)
        if not embeddings:
            print("❌ 임베딩 생성 실패")
            return False
        
        # 5단계: HNSW 인덱스 생성
        success = self.step4_create_hnsw_index(chunked_docs, embeddings)
        
        if success:
            print("\n🎉 전체 FAISS HNSW 시스템 구축 완료!")
            self._print_system_statistics()
        
        return success
    
    def _print_system_statistics(self):
        """시스템 통계 정보 출력"""
        if not self.faiss_index:
            return
        
        print("\n📊 시스템 통계:")
        print("-" * 30)
        
        # 기본 정보
        print(f"   • 임베딩 모델: {self.embedding_type}")
        print(f"   • 인덱스 타입: FAISS HNSW")
        print(f"   • 저장 위치: {self.index_dir}")
        
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
    
    def interactive_demo(self):
        """대화형 데모"""
        print("\n🎓 교육용 대화형 데모")
        print("=" * 50)
        print("사용 가능한 명령어:")
        print("  - 검색어만 입력: 직접 검색 (예: 'RSI', '볼린저밴드')")
        print("  - 'search <검색어>': 명시적 검색 명령")
        print("  - 'stats': 시스템 통계")
        print("  - 'quit': 종료")
        print(f"\n현재 임베딩 모델: {self.embedding_type}")
        print()
        
        while True:
            try:
                command = input("검색 명령어 입력: ").strip()
                
                if command.lower() == 'quit':
                    break
                elif command.lower() == 'stats':
                    self._print_system_statistics()
                elif command.startswith('search '):
                    # 명시적 search 명령어 처리
                    query = command[7:].strip()
                    if query:
                        results = self.step5_similarity_search(query)
                        if results:
                            self._save_search_results(results, query)
                    else:
                        print("   ❌ 검색어를 입력해주세요.")
                elif command:
                    # 단순 검색어로 처리 (search 접두사 없이)
                    results = self.step5_similarity_search(command)
                    if results:
                        self._save_search_results(results, command)
                    else:
                        print("   ❌ 검색 결과가 없습니다.")
                else:
                    print("   ❌ 검색어를 입력해주세요.")
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 데모를 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")
    
    def _save_search_results(self, results: List[Tuple[Document, float]], query: str):
        """검색 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"educational_search_results_{timestamp}.json"
        
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
    
    print(f"🎓 교육용 FAISS HNSW 시스템 - 임베딩 모델: {embedding_type}")
    
    # 교육용 FAISS 시스템 초기화
    educational_system = EducationalFAISSSystem(embedding_type=embedding_type)
    
    # 전체 시스템 구축
    if educational_system.build_complete_system():
        print("\n🎉 교육용 FAISS HNSW 시스템 구축 완료!")
        
        # 대화형 데모 실행
        educational_system.interactive_demo()
        
    else:
        print("❌ 시스템 구축에 실패했습니다.")

if __name__ == "__main__":
    main() 