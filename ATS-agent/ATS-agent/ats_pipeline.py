"""
ATS Resume Analyzer Pipeline
Main pipeline for analyzing resume-job description similarity
"""

import json
from typing import Dict, Optional
from pdf_extractor import PDFExtractor
from keyword_extractor import KeywordExtractor
from similarity_calculator import SimilarityCalculator


class ATSPipeline:
    """
    Complete ATS Pipeline for resume analysis
    
    This pipeline:
    1. Extracts text from PDF resume
    2. Extracts keywords from resume and job description
    3. Calculates similarity scores
    4. Provides recommendations
    """
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize the ATS pipeline
        
        Args:
            use_spacy: Whether to use spaCy for advanced NLP
        """
        self.pdf_extractor = PDFExtractor()
        self.keyword_extractor = KeywordExtractor(use_spacy=use_spacy)
        self.similarity_calculator = SimilarityCalculator()
    
    def analyze(self, resume_pdf_path: str, job_description: str, 
                verbose: bool = True, analyze_format: bool = True) -> Dict:
        """
        Analyze resume against job description
        
        Args:
            resume_pdf_path: Path to the resume PDF file
            job_description: Job description text
            verbose: Whether to print progress messages
            analyze_format: Whether to analyze CV format and structure
            
        Returns:
            Dictionary containing complete analysis results
        """
        if verbose:
            print("🔍 Starting ATS Analysis...")
        
        # Step 0: Analyze PDF format (if requested)
        format_analysis = None
        if analyze_format:
            if verbose:
                print("📋 Analyzing CV format and structure...")
            
            format_analysis = self.pdf_extractor.analyze_pdf(resume_pdf_path)
            
            if format_analysis and format_analysis.get('success'):
                if verbose:
                    structure = format_analysis['structure_analysis']
                    print(f"   CV Format: {structure['cv_format']}")
                    print(f"   Sections Detected: {structure['section_count']}")
                    print(f"   ATS Score: {structure['ats_friendly_score']}/100 - {structure['ats_friendly_rating']}")
        
        # Step 1: Extract text from resume PDF
        if verbose:
            print("📄 Extracting text from resume PDF...")
        
        try:
            resume_text = self.pdf_extractor.extract_text(resume_pdf_path)
            if not resume_text:
                return {
                    'error': 'Failed to extract text from resume PDF',
                    'success': False,
                    'format_analysis': format_analysis
                }
        except Exception as e:
            return {
                'error': f'Error reading PDF: {str(e)}',
                'success': False,
                'format_analysis': format_analysis
            }
        
        if verbose:
            print(f"   Extracted {len(resume_text)} characters from resume")
        
        # Step 2: Extract keywords from resume
        if verbose:
            print("🔑 Extracting keywords from resume...")
        
        resume_keywords = self.keyword_extractor.extract_keywords(resume_text)
        
        if verbose:
            print(f"   Found {len(resume_keywords.get('all_keywords', []))} unique keywords")
            print(f"   Detected {len(resume_keywords.get('technical_skills', []))} technical skills")
        
        # Step 3: Extract keywords from job description
        if verbose:
            print("🔑 Extracting keywords from job description...")
        
        job_keywords = self.keyword_extractor.extract_keywords(job_description)
        
        if verbose:
            print(f"   Found {len(job_keywords.get('all_keywords', []))} unique keywords")
            print(f"   Detected {len(job_keywords.get('technical_skills', []))} technical skills")
        
        # Step 4: Calculate similarity scores
        if verbose:
            print("📊 Calculating similarity scores...")
        
        similarity_results = self.similarity_calculator.calculate_weighted_score(
            resume_text,
            job_description,
            resume_keywords,
            job_keywords
        )
        
        if verbose:
            print(f"   Overall Match Score: {similarity_results['overall_percentage']}%")
            print(f"   Match Level: {similarity_results['match_level']}")
        
        # Step 5: Generate recommendations
        recommendations = self.similarity_calculator.generate_recommendations(similarity_results)
        
        # Compile complete results
        results = {
            'success': True,
            'format_analysis': format_analysis,
            'resume_analysis': {
                'text_length': len(resume_text),
                'keywords': resume_keywords,
                'technical_skills': resume_keywords.get('technical_skills', [])
            },
            'job_analysis': {
                'text_length': len(job_description),
                'keywords': job_keywords,
                'technical_skills': job_keywords.get('technical_skills', [])
            },
            'similarity_scores': similarity_results,
            'recommendations': recommendations
        }
        
        if verbose:
            print("\n" + "="*60)
            print("📈 ANALYSIS COMPLETE")
            print("="*60)
            self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict):
        """
        Print a formatted summary of the analysis results
        
        Args:
            results: Analysis results dictionary
        """
        if not results.get('success'):
            print(f"❌ Error: {results.get('error', 'Unknown error')}")
            return
        
        similarity = results['similarity_scores']
        
        print(f"\n🎯 Overall Match Score: {similarity['overall_percentage']}%")
        print(f"📊 Match Level: {similarity['match_level']}")
        
        print(f"\n📋 Detailed Scores:")
        print(f"   • Text Similarity: {similarity['detailed_scores']['text_similarity']:.2%}")
        print(f"   • Skills Match: {similarity['detailed_scores']['skills_match_rate']:.2%}")
        print(f"   • Keywords Match: {similarity['detailed_scores']['tfidf_match_rate']:.2%}")
        
        print(f"\n✅ Matched Skills ({len(similarity['matched_skills'])}):")
        if similarity['matched_skills']:
            for skill in similarity['matched_skills'][:10]:
                print(f"   • {skill}")
            if len(similarity['matched_skills']) > 10:
                print(f"   ... and {len(similarity['matched_skills']) - 10} more")
        else:
            print("   None")
        
        print(f"\n❌ Missing Skills ({len(similarity['missing_skills'])}):")
        if similarity['missing_skills']:
            for skill in similarity['missing_skills'][:10]:
                print(f"   • {skill}")
            if len(similarity['missing_skills']) > 10:
                print(f"   ... and {len(similarity['missing_skills']) - 10} more")
        else:
            print("   None")
        
        print(f"\n💡 Recommendations:")
        for rec in results['recommendations']:
            print(f"   {rec}")
    
    def save_results(self, results: Dict, output_path: str):
        """
        Save analysis results to a JSON file
        
        Args:
            results: Analysis results dictionary
            output_path: Path to save the JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to: {output_path}")


def main():
    """Example usage of the ATS Pipeline"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python ats_pipeline.py <resume.pdf> <job_description.txt>")
        print("\nExample:")
        print("  python ats_pipeline.py resume.pdf job_description.txt")
        sys.exit(1)
    
    resume_path = sys.argv[1]
    job_desc_path = sys.argv[2]
    
    # Read job description
    try:
        with open(job_desc_path, 'r', encoding='utf-8') as f:
            job_description = f.read()
    except Exception as e:
        print(f"❌ Error reading job description: {e}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = ATSPipeline(use_spacy=True)
    results = pipeline.analyze(resume_path, job_description, verbose=True)
    
    # Save results
    if results.get('success'):
        pipeline.save_results(results, 'ats_analysis_results.json')


if __name__ == "__main__":
    main()
