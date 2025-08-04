"""
FAISS RAG 시스템 테스트 스크립트
RAG 시스템의 기능을 테스트하고 성능을 확인
"""

from faiss_rag_system import FAISSRAGSystem

def test_rag_system():
    """RAG 시스템 테스트"""
    print("🤖 FAISS RAG 시스템 테스트 시작")
    print("=" * 60)
    
    # RAG 시스템 초기화 (Gemini 사용)
    rag_system = FAISSRAGSystem(embedding_type="gemini")
    
    # 인덱스 로드
    if not rag_system.load_index():
        print("❌ 인덱스 로드 실패")
        return
    
    print("✅ RAG 시스템 준비 완료")
    print()
    
    # 테스트 질문들
    test_questions = [
        "RSI란 무엇인가요?",
        "볼린저밴드 사용법을 알려주세요",
        "MACD 지표의 의미는?",
        "기술적 분석의 기본 원리는?",
        "엘리어트 파동이론이란?"
    ]
    
    # 각 질문으로 테스트
    for i, question in enumerate(test_questions, 1):
        print(f"🔍 테스트 {i}/{len(test_questions)}: '{question}'")
        print("-" * 60)
        
        result = rag_system.search_and_generate_answer(question, k=3)
        
        if result and result.get('answer'):
            print(f"✅ 답변 생성 성공")
            print(f"📄 검색된 문서: {len(result.get('search_results', []))}개")
            print(f"🤖 답변 길이: {len(result['answer'])}자")
            print()
            
            # 답변 미리보기
            answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
            print(f"답변 미리보기: {answer_preview}")
            print()
            
            # 결과 저장
            rag_system.save_rag_results(result)
            
        else:
            print("❌ 답변 생성 실패")
        
        print("=" * 60)
        print()
    
    print("🎉 RAG 시스템 테스트 완료")

if __name__ == "__main__":
    test_rag_system() 