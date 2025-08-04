"""
교육용 FAISS HNSW 시스템 테스트 스크립트
각 단계별로 테스트를 수행하여 시스템이 올바르게 작동하는지 확인
"""

import sys
from pathlib import Path
from educational_faiss_system import EducationalFAISSSystem

def test_step_by_step():
    """단계별 테스트 수행"""
    print("🧪 교육용 FAISS HNSW 시스템 단계별 테스트")
    print("=" * 60)
    
    # 시스템 초기화
    print("\n1️⃣ 시스템 초기화 테스트")
    try:
        system = EducationalFAISSSystem(embedding_type="huggingface")
        print("   ✅ 시스템 초기화 성공")
    except Exception as e:
        print(f"   ❌ 시스템 초기화 실패: {str(e)}")
        return False
    
    # 2단계: 문서 로드 테스트
    print("\n2️⃣ 문서 로드 테스트")
    try:
        documents = system.step1_load_documents()
        if documents:
            print(f"   ✅ 문서 로드 성공: {len(documents)}개 문서")
        else:
            print("   ❌ 문서 로드 실패: 문서가 없습니다")
            return False
    except Exception as e:
        print(f"   ❌ 문서 로드 실패: {str(e)}")
        return False
    
    # 3단계: 토크나이징 및 청킹 테스트
    print("\n3️⃣ 토크나이징 및 청킹 테스트")
    try:
        chunked_docs = system.step2_tokenize_and_chunk(documents)
        if chunked_docs:
            print(f"   ✅ 청킹 성공: {len(chunked_docs)}개 청크")
        else:
            print("   ❌ 청킹 실패: 청크가 생성되지 않았습니다")
            return False
    except Exception as e:
        print(f"   ❌ 청킹 실패: {str(e)}")
        return False
    
    # 4단계: 임베딩 생성 테스트
    print("\n4️⃣ 임베딩 생성 테스트")
    try:
        embeddings = system.step3_create_embeddings(chunked_docs)
        if embeddings:
            print(f"   ✅ 임베딩 생성 성공: {len(embeddings)}개 벡터")
            print(f"   📏 벡터 차원: {len(embeddings[0])}")
        else:
            print("   ❌ 임베딩 생성 실패: 벡터가 생성되지 않았습니다")
            return False
    except Exception as e:
        print(f"   ❌ 임베딩 생성 실패: {str(e)}")
        return False
    
    # 5단계: HNSW 인덱스 생성 테스트
    print("\n5️⃣ HNSW 인덱스 생성 테스트")
    try:
        success = system.step4_create_hnsw_index(chunked_docs, embeddings)
        if success:
            print("   ✅ HNSW 인덱스 생성 성공")
        else:
            print("   ❌ HNSW 인덱스 생성 실패")
            return False
    except Exception as e:
        print(f"   ❌ HNSW 인덱스 생성 실패: {str(e)}")
        return False
    
    # 6단계: 검색 테스트
    print("\n6️⃣ 검색 테스트")
    try:
        test_queries = [
            "RSI 지표",
            "볼린저 밴드",
            "이동평균선",
            "기술적 분석"
        ]
        
        for query in test_queries:
            results = system.step5_similarity_search(query, k=2)
            if results:
                print(f"   ✅ '{query}' 검색 성공: {len(results)}개 결과")
            else:
                print(f"   ⚠️ '{query}' 검색 결과 없음")
    except Exception as e:
        print(f"   ❌ 검색 테스트 실패: {str(e)}")
        return False
    
    print("\n🎉 모든 테스트 통과!")
    return True

def test_complete_system():
    """전체 시스템 테스트"""
    print("\n🚀 전체 시스템 테스트")
    print("=" * 40)
    
    try:
        system = EducationalFAISSSystem(embedding_type="huggingface")
        
        # 전체 시스템 구축
        success = system.build_complete_system()
        
        if success:
            print("   ✅ 전체 시스템 구축 성공")
            
            # 간단한 검색 테스트
            results = system.step5_similarity_search("RSI", k=1)
            if results:
                print("   ✅ 검색 기능 정상 작동")
                return True
            else:
                print("   ⚠️ 검색 결과가 없습니다")
                return True  # 검색 결과가 없는 것은 오류가 아님
        else:
            print("   ❌ 전체 시스템 구축 실패")
            return False
            
    except Exception as e:
        print(f"   ❌ 전체 시스템 테스트 실패: {str(e)}")
        return False

def test_interactive_mode():
    """대화형 모드 테스트"""
    print("\n💬 대화형 모드 테스트")
    print("=" * 30)
    print("이 테스트는 수동으로 진행됩니다.")
    print("시스템이 정상적으로 초기화되면 대화형 인터페이스가 시작됩니다.")
    print("'quit'를 입력하여 종료할 수 있습니다.")
    
    try:
        system = EducationalFAISSSystem(embedding_type="huggingface")
        
        if system.build_complete_system():
            print("   ✅ 시스템 준비 완료")
            print("   🔄 대화형 인터페이스 시작...")
            system.interactive_demo()
            return True
        else:
            print("   ❌ 시스템 구축 실패")
            return False
            
    except Exception as e:
        print(f"   ❌ 대화형 모드 테스트 실패: {str(e)}")
        return False

def main():
    """메인 테스트 함수"""
    print("🎓 교육용 FAISS HNSW 시스템 테스트 스위트")
    print("=" * 60)
    
    # 테스트 모드 선택
    if len(sys.argv) > 1:
        test_mode = sys.argv[1].lower()
    else:
        test_mode = "step"
    
    success = False
    
    if test_mode == "step":
        success = test_step_by_step()
    elif test_mode == "complete":
        success = test_complete_system()
    elif test_mode == "interactive":
        success = test_interactive_mode()
    else:
        print("❌ 잘못된 테스트 모드입니다.")
        print("사용 가능한 모드: step, complete, interactive")
        return
    
    if success:
        print("\n🎉 테스트 완료!")
        print("시스템이 정상적으로 작동합니다.")
    else:
        print("\n❌ 테스트 실패!")
        print("문제를 확인하고 다시 시도해주세요.")

if __name__ == "__main__":
    main() 