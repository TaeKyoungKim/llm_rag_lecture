"""
교육용 FAISS HNSW 시스템 실행 예시
간단한 예시를 통해 시스템 사용법을 보여줍니다.
"""

from educational_faiss_system import EducationalFAISSSystem

def simple_example():
    """간단한 실행 예시"""
    print("🎓 교육용 FAISS HNSW 시스템 - 간단한 예시")
    print("=" * 60)
    
    # 1. 시스템 초기화
    print("\n1️⃣ 시스템 초기화")
    system = EducationalFAISSSystem(embedding_type="huggingface")
    print("   ✅ 시스템 초기화 완료")
    
    # 2. 전체 시스템 구축
    print("\n2️⃣ 전체 시스템 구축")
    if system.build_complete_system():
        print("   ✅ 시스템 구축 완료")
    else:
        print("   ❌ 시스템 구축 실패")
        return
    
    # 3. 간단한 검색 예시
    print("\n3️⃣ 검색 예시")
    
    # RSI 관련 검색
    print("\n   🔍 RSI 관련 검색:")
    rsi_results = system.step5_similarity_search("RSI 지표 활용법", k=2)
    if rsi_results:
        for i, (doc, score) in enumerate(rsi_results, 1):
            print(f"      결과 {i}: 유사도 {score:.4f}")
            print(f"         페이지: {doc.metadata.get('page', 'N/A')}")
            print(f"         내용: {doc.page_content[:80]}...")
    
    # 볼린저 밴드 관련 검색
    print("\n   🔍 볼린저 밴드 관련 검색:")
    bb_results = system.step5_similarity_search("볼린저 밴드 분석", k=2)
    if bb_results:
        for i, (doc, score) in enumerate(bb_results, 1):
            print(f"      결과 {i}: 유사도 {score:.4f}")
            print(f"         페이지: {doc.metadata.get('page', 'N/A')}")
            print(f"         내용: {doc.page_content[:80]}...")
    
    # 4. 시스템 통계 출력
    print("\n4️⃣ 시스템 통계")
    system._print_system_statistics()
    
    print("\n🎉 예시 실행 완료!")

def advanced_example():
    """고급 실행 예시"""
    print("\n🚀 교육용 FAISS HNSW 시스템 - 고급 예시")
    print("=" * 60)
    
    # 1. 시스템 초기화 (Gemini 임베딩 사용)
    print("\n1️⃣ Gemini 임베딩으로 시스템 초기화")
    try:
        system = EducationalFAISSSystem(embedding_type="gemini")
        print("   ✅ Gemini 임베딩 시스템 초기화 완료")
    except Exception as e:
        print(f"   ⚠️ Gemini 임베딩 실패, HuggingFace로 대체: {str(e)}")
        system = EducationalFAISSSystem(embedding_type="huggingface")
        print("   ✅ HuggingFace 임베딩 시스템 초기화 완료")
    
    # 2. 단계별 실행
    print("\n2️⃣ 단계별 실행")
    
    # 문서 로드
    print("   📄 문서 로드 중...")
    documents = system.step1_load_documents()
    if not documents:
        print("   ❌ 문서 로드 실패")
        return
    print(f"   ✅ {len(documents)}개 문서 로드 완료")
    
    # 청킹
    print("   ✂️ 문서 청킹 중...")
    chunked_docs = system.step2_tokenize_and_chunk(documents)
    if not chunked_docs:
        print("   ❌ 문서 청킹 실패")
        return
    print(f"   ✅ {len(chunked_docs)}개 청크 생성 완료")
    
    # 임베딩
    print("   🔢 임베딩 생성 중...")
    embeddings = system.step3_create_embeddings(chunked_docs)
    if not embeddings:
        print("   ❌ 임베딩 생성 실패")
        return
    print(f"   ✅ {len(embeddings)}개 임베딩 생성 완료")
    
    # HNSW 인덱스 생성
    print("   🔍 HNSW 인덱스 생성 중...")
    success = system.step4_create_hnsw_index(chunked_docs, embeddings)
    if not success:
        print("   ❌ HNSW 인덱스 생성 실패")
        return
    print("   ✅ HNSW 인덱스 생성 완료")
    
    # 3. 다양한 검색 예시
    print("\n3️⃣ 다양한 검색 예시")
    
    search_queries = [
        ("RSI 과매수 과매도", "RSI 지표의 과매수/과매도 판단"),
        ("MACD 골든크로스", "MACD 골든크로스 신호"),
        ("볼린저 밴드 변동성", "볼린저 밴드의 변동성 분석"),
        ("이동평균선 추세", "이동평균선을 이용한 추세 분석"),
        ("스토캐스틱 오실레이터", "스토캐스틱 오실레이터 활용")
    ]
    
    for query, description in search_queries:
        print(f"\n   🔍 {description}:")
        results = system.step5_similarity_search(query, k=1)
        if results:
            doc, score = results[0]
            print(f"      최고 유사도: {score:.4f}")
            print(f"      페이지: {doc.metadata.get('page', 'N/A')}")
            print(f"      기술적 내용 포함: {'✅' if doc.metadata.get('has_technical_content', False) else '❌'}")
            print(f"      내용 미리보기: {doc.page_content[:100]}...")
        else:
            print(f"      ⚠️ 검색 결과 없음")
    
    # 4. 검색 결과 저장
    print("\n4️⃣ 검색 결과 저장")
    for query, description in search_queries[:2]:  # 처음 2개만 저장
        results = system.step5_similarity_search(query, k=3)
        if results:
            system._save_search_results(results, query)
    
    print("\n🎉 고급 예시 실행 완료!")

def main():
    """메인 실행 함수"""
    import sys
    
    # 실행 모드 선택
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "simple"
    
    if mode == "simple":
        simple_example()
    elif mode == "advanced":
        advanced_example()
    elif mode == "both":
        simple_example()
        advanced_example()
    else:
        print("❌ 잘못된 실행 모드입니다.")
        print("사용 가능한 모드: simple, advanced, both")
        print("\n예시:")
        print("  python run_educational_example.py simple")
        print("  python run_educational_example.py advanced")
        print("  python run_educational_example.py both")

if __name__ == "__main__":
    main() 