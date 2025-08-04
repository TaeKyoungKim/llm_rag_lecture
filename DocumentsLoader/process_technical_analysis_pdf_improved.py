# 개선된 실제 PDF 파일 처리 시스템 - 한국어 기술적 분석 용어 최적화

# 필요한 패키지 설치
# uv add langchain_community pypdf rapidocr-onnxruntime pytesseract pandas requests

import os
import re
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

# LangChain PDF 로더
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_community.document_loaders.parsers import RapidOCRBlobParser, TesseractBlobParser

print("🔧 개선된 실제 PDF 파일 처리 시스템 초기화")
print("=" * 60)

class ImprovedPDFProcessor:
    """개선된 실제 PDF 파일을 처리하는 클래스"""
    
    def __init__(self):
        self.pdf_path = Path("DocumentsLoader/data/기술적차트분석이론및방법.pdf")
        
        # 한국어 기술적 분석 키워드 (더 포괄적)
        self.technical_keywords = {
            'indicators': [
                # 기본 지표
                'RSI', '상대강도지수', 'MACD', '이동평균수렴확산',
                '볼린저밴드', 'Bollinger', '스토캐스틱', 'Stochastic',
                'ATR', 'CCI', 'Williams %R', '이동평균', 'Moving Average',
                'KDJ', 'DMI', 'OBV', 'VR', 'ROC', 'MOM', 'ADX',
                # 한국어 변형
                '이동평균선', '이평선', '이동평균', '이평', '이동평균수렴',
                '상대강도', '상대강도지수', 'RSI지수', 'RSI지표',
                '볼린저', '볼린저밴드', '볼린저밴드지표',
                '스토캐스틱', '스토캐스틱지표', '스토캐스틱오실레이터',
                '파라볼릭', 'Parabolic', 'SAR', 'Parabolic SAR',
                '일목균형표', '일목균형', '균형표',
                '피보나치', 'Fibonacci', '피보나치되돌림', '피보나치확장',
                '엘리어트', 'Elliott', '엘리어트파동', '파동이론'
            ],
            'patterns': [
                # 기본 패턴
                '삼각형', '쐐기형', '플래그', '헤드앤숄더', '더블탑', '더블바텀',
                '컵앤핸들', '역헤드앤숄더', '상승삼각형', '하락삼각형',
                'Triangle', 'Wedge', 'Flag', 'Head and Shoulders', 'Double Top', 'Double Bottom',
                'Cup and Handle', 'Inverse Head and Shoulders',
                # 한국어 변형
                '삼각형패턴', '쐐기형패턴', '플래그패턴', '헤드앤숄더패턴',
                '더블탑패턴', '더블바텀패턴', '컵앤핸들패턴',
                '일반패턴', '지속패턴', '반전패턴', '차트패턴'
            ],
            'concepts': [
                # 기본 개념
                '지지선', '저항선', '추세선', '과매수', '과매도', '다이버전스',
                '크로스오버', '브레이크아웃', '풀백', '되돌림', '골든크로스', '데드크로스',
                'Support', 'Resistance', 'Trend', 'Overbought', 'Oversold', 'Divergence',
                'Crossover', 'Breakout', 'Pullback', 'Retracement',
                # 한국어 변형
                '지지', '저항', '추세', '추세대', '추세선',
                '과매수', '과매도', '오버바이', '오버솔드',
                '다이버전스', '다이버전스이해', '다이버전스종류',
                '크로스', '크로스오버', '골든크로스', '데드크로스',
                '브레이크아웃', '브레이크', '돌파', '돌파선',
                '풀백', '되돌림', '조정', '반등',
                '거래량', '거래량기초', '거래량과시세', '거래량으로보는세력활동',
                '매물대', '매물', '매물대분석',
                '포트폴리오', '포트폴리오이론',
                '김치프리미엄', '역프리미엄'
            ],
            'levels': [
                # 기본 수치
                '30', '70', '50', '20', '80', '0.618', '0.382', '1.618',
                '0.236', '0.5', '0.786', '1.0', '1.272', '1.414', '2.0',
                # 한국어 변형
                '30%', '70%', '50%', '20%', '80%',
                '0.618', '0.382', '1.618', '황금비율',
                '피보나치레벨', '피보나치수열', '황금비율'
            ]
        }
    
    def check_pdf_exists(self) -> bool:
        """PDF 파일 존재 여부 확인"""
        if self.pdf_path.exists():
            print(f"✅ PDF 파일 발견: {self.pdf_path}")
            print(f"   파일 크기: {self.pdf_path.stat().st_size / 1024:.1f} KB")
            return True
        else:
            print(f"❌ PDF 파일을 찾을 수 없습니다: {self.pdf_path}")
            return False
    
    def load_pdf_with_pypdf(self) -> Optional[List]:
        """PyPDFLoader를 사용하여 PDF 로드"""
        try:
            print(f"\n📄 PyPDFLoader로 PDF 로드 중...")
            print(f"   파일 경로: {self.pdf_path}")
            
            loader = PyPDFLoader(str(self.pdf_path))
            docs = loader.load()
            
            print(f"   ✅ PDF 로드 성공!")
            print(f"   📊 문서 정보:")
            print(f"      - 총 문서 수: {len(docs)}개")
            
            for i, doc in enumerate(docs[:3]):  # 처음 3개 문서만 표시
                print(f"      - 문서 {i+1}: {len(doc.page_content)}자")
                if doc.metadata:
                    print(f"        메타데이터: {doc.metadata}")
            
            if len(docs) > 3:
                print(f"      - ... 외 {len(docs) - 3}개 문서")
            
            return docs
            
        except Exception as e:
            print(f"   ❌ PDF 로드 실패: {str(e)}")
            return None
    
    def extract_technical_indicators_improved(self, content: str) -> Dict[str, Any]:
        """개선된 기술적 분석 지표 추출"""
        extracted_info = {
            'indicators_found': [],
            'patterns_found': [],
            'concepts_found': [],
            'numeric_levels': [],
            'analysis_summary': {}
        }
        
        content_lower = content.lower()
        
        # 지표 검색 (더 정확한 매칭)
        for category, keywords in self.technical_keywords.items():
            found_items = []
            for keyword in keywords:
                # 정확한 단어 매칭 (부분 문자열이 아닌)
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, content_lower) or keyword in content:
                    found_items.append(keyword)
            
            if category == 'levels':
                extracted_info['numeric_levels'] = found_items
            else:
                extracted_info[f'{category[:-1]}_found'] = found_items
        
        # 추가 패턴 검색 (정규표현식 사용)
        additional_patterns = {
            'indicators': [
                r'RSI\s*\([^)]*\)',  # RSI (Relative Strength Index)
                r'MACD\s*\([^)]*\)',  # MACD (...)
                r'볼린저\s*밴드',  # 볼린저 밴드
                r'이동평균\s*선',  # 이동평균 선
                r'스토캐스틱',  # 스토캐스틱
                r'파라볼릭\s*SAR',  # 파라볼릭 SAR
                r'일목\s*균형표',  # 일목 균형표
                r'피보나치\s*되돌림',  # 피보나치 되돌림
                r'엘리어트\s*파동'  # 엘리어트 파동
            ],
            'concepts': [
                r'지지\s*선',  # 지지 선
                r'저항\s*선',  # 저항 선
                r'추세\s*선',  # 추세 선
                r'과매수',  # 과매수
                r'과매도',  # 과매도
                r'다이버전스',  # 다이버전스
                r'골든\s*크로스',  # 골든 크로스
                r'데드\s*크로스',  # 데드 크로스
                r'거래량',  # 거래량
                r'매물대'  # 매물대
            ]
        }
        
        # 추가 패턴 검색
        for category, patterns in additional_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if category == 'indicators' and match not in extracted_info['indicators_found']:
                        extracted_info['indicators_found'].append(match)
                    elif category == 'concepts' and match not in extracted_info['concepts_found']:
                        extracted_info['concepts_found'].append(match)
        
        # 분석 요약 생성
        total_indicators = len(extracted_info['indicators_found'])
        total_patterns = len(extracted_info['patterns_found'])
        total_concepts = len(extracted_info['concepts_found'])
        
        extracted_info['analysis_summary'] = {
            'total_indicators': total_indicators,
            'total_patterns': total_patterns,
            'total_concepts': total_concepts,
            'content_quality_score': min(100, (total_indicators * 10) + (total_patterns * 5) + (total_concepts * 3)),
            'main_focus': self._determine_main_focus(extracted_info)
        }
        
        return extracted_info
    
    def _determine_main_focus(self, extracted_info: Dict) -> str:
        """주요 포커스 결정"""
        indicators = extracted_info['indicators_found']
        concepts = extracted_info['concepts_found']
        
        # RSI 관련
        if any('RSI' in ind or '상대강도' in ind for ind in indicators):
            if any('MACD' in ind or '이동평균수렴' in ind for ind in indicators):
                return "RSI + MACD 복합 분석"
            return "RSI 중심 분석"
        
        # MACD 관련
        elif any('MACD' in ind or '이동평균수렴' in ind for ind in indicators):
            return "MACD 중심 분석"
        
        # 볼린저밴드 관련
        elif any('볼린저' in ind or 'Bollinger' in ind for ind in indicators):
            return "볼린저밴드 중심 분석"
        
        # 이동평균선 관련
        elif any('이동평균' in ind or '이평' in ind for ind in indicators):
            return "이동평균선 중심 분석"
        
        # 엘리어트 파동 관련
        elif any('엘리어트' in ind or '파동' in ind for ind in indicators):
            return "엘리어트 파동 이론"
        
        # 일목균형표 관련
        elif any('일목' in ind or '균형표' in ind for ind in indicators):
            return "일목균형표 분석"
        
        # 피보나치 관련
        elif any('피보나치' in ind for ind in indicators):
            return "피보나치 되돌림 분석"
        
        # 거래량 관련
        elif any('거래량' in concept for concept in concepts):
            return "거래량 분석"
        
        # 추세 관련
        elif any('추세' in concept for concept in concepts):
            return "추세 분석"
        
        else:
            return "일반 기술적 분석"
    
    def analyze_pdf_content(self, docs: List) -> Dict[str, Any]:
        """PDF 내용 종합 분석"""
        print(f"\n🔍 PDF 내용 분석 중...")
        
        # 모든 문서 내용 합치기
        full_content = ""
        for i, doc in enumerate(docs):
            full_content += f"\n--- 페이지 {i+1} ---\n"
            full_content += doc.page_content
            full_content += "\n"
        
        print(f"   📝 총 텍스트 길이: {len(full_content):,}자")
        
        # 개선된 기술적 분석 지표 추출
        analysis_results = self.extract_technical_indicators_improved(full_content)
        
        return {
            'full_content': full_content,
            'analysis_results': analysis_results,
            'document_count': len(docs)
        }
    
    def generate_detailed_report(self, analysis_data: Dict):
        """상세 분석 리포트 생성"""
        print(f"\n📈 기술적 분석 PDF 상세 리포트")
        print("=" * 60)
        
        # 기본 정보
        print(f"\n📄 PDF 기본 정보:")
        print(f"   • 파일명: {self.pdf_path.name}")
        print(f"   • 문서 수: {analysis_data['document_count']}개")
        print(f"   • 총 텍스트 길이: {len(analysis_data['full_content']):,}자")
        
        # 기술적 분석 내용 추출 결과
        analysis_results = analysis_data['analysis_results']
        summary = analysis_results['analysis_summary']
        
        print(f"\n🔍 기술적 분석 내용 추출 결과:")
        print(f"   • 발견된 기술 지표: {summary['total_indicators']}개")
        if analysis_results['indicators_found']:
            print(f"     → {', '.join(analysis_results['indicators_found'][:10])}")
            if len(analysis_results['indicators_found']) > 10:
                print(f"     → ... 외 {len(analysis_results['indicators_found']) - 10}개")
        
        print(f"   • 발견된 패턴: {summary['total_patterns']}개")
        if analysis_results['patterns_found']:
            print(f"     → {', '.join(analysis_results['patterns_found'])}")
        
        print(f"   • 발견된 개념: {summary['total_concepts']}개")
        if analysis_results['concepts_found']:
            print(f"     → {', '.join(analysis_results['concepts_found'][:10])}")
            if len(analysis_results['concepts_found']) > 10:
                print(f"     → ... 외 {len(analysis_results['concepts_found']) - 10}개")
        
        print(f"   • 주요 포커스: {summary['main_focus']}")
        print(f"   • 내용 품질 점수: {summary['content_quality_score']}/100")
        
        # 품질 등급
        score = summary['content_quality_score']
        if score >= 80:
            grade = "🥇 최우수 (전문 수준)"
        elif score >= 60:
            grade = "🥈 우수 (실용 수준)"
        elif score >= 40:
            grade = "🥉 양호 (기초 수준)"
        else:
            grade = "📝 기본 (입문 수준)"
        
        print(f"   • 품질 등급: {grade}")
        
        # 내용 미리보기
        print(f"\n📖 내용 미리보기 (처음 500자):")
        preview = analysis_data['full_content'][:500]
        print(f"   {preview}...")
        
        return analysis_results
    
    def save_analysis_results(self, analysis_data: Dict, output_file: str = "improved_technical_analysis_results.txt"):
        """분석 결과를 파일로 저장"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("개선된 기술적 분석 PDF 처리 결과\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"파일명: {self.pdf_path.name}\n")
                f.write(f"처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"문서 수: {analysis_data['document_count']}개\n")
                f.write(f"총 텍스트 길이: {len(analysis_data['full_content']):,}자\n\n")
                
                analysis_results = analysis_data['analysis_results']
                summary = analysis_results['analysis_summary']
                
                f.write("발견된 기술 지표:\n")
                for indicator in analysis_results['indicators_found']:
                    f.write(f"  - {indicator}\n")
                f.write("\n")
                
                f.write("발견된 패턴:\n")
                for pattern in analysis_results['patterns_found']:
                    f.write(f"  - {pattern}\n")
                f.write("\n")
                
                f.write("발견된 개념:\n")
                for concept in analysis_results['concepts_found']:
                    f.write(f"  - {concept}\n")
                f.write("\n")
                
                f.write(f"주요 포커스: {summary['main_focus']}\n")
                f.write(f"품질 점수: {summary['content_quality_score']}/100\n\n")
                
                f.write("전체 내용:\n")
                f.write("-" * 30 + "\n")
                f.write(analysis_data['full_content'])
            
            print(f"   ✅ 분석 결과가 {output_file}에 저장되었습니다.")
            
        except Exception as e:
            print(f"   ❌ 파일 저장 실패: {str(e)}")
    
    def process_pdf(self):
        """PDF 처리 메인 함수"""
        print("🚀 개선된 PDF 처리 시작")
        print("=" * 60)
        
        # 1. PDF 파일 존재 확인
        if not self.check_pdf_exists():
            return None
        
        # 2. PyPDFLoader로 PDF 로드
        docs = self.load_pdf_with_pypdf()
        if not docs:
            print("❌ PDF 로드에 실패했습니다.")
            return None
        
        # 3. PDF 내용 분석
        analysis_data = self.analyze_pdf_content(docs)
        
        # 4. 상세 리포트 생성
        analysis_results = self.generate_detailed_report(analysis_data)
        
        # 5. 결과 저장
        self.save_analysis_results(analysis_data)
        
        return analysis_data

def main():
    """메인 실행 함수"""
    processor = ImprovedPDFProcessor()
    result = processor.process_pdf()
    
    if result:
        print(f"\n🎉 개선된 PDF 처리 완료!")
        print(f"   📊 총 {result['document_count']}개 문서 처리")
        print(f"   📝 {len(result['full_content']):,}자 텍스트 추출")
        print(f"   🔍 {result['analysis_results']['analysis_summary']['total_indicators']}개 기술 지표 발견")
        print(f"   🎯 {result['analysis_results']['analysis_summary']['main_focus']}")
    else:
        print(f"\n❌ PDF 처리에 실패했습니다.")

if __name__ == "__main__":
    main() 