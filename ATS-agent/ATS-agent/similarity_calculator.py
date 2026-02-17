"""
Similarity Calculation Module
Calculates similarity scores between resume and job description
"""

from typing import List, Dict, Tuple
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityCalculator:
    """Calculate similarity between resume and job description"""
    
    def __init__(self):
        pass
    
    def jaccard_similarity(self, set1: set, set2: set) -> float:
        """
        Calculate Jaccard similarity between two sets
        
        Args:
            set1: First set of keywords
            set2: Second set of keywords
            
        Returns:
            Jaccard similarity score (0 to 1)
        """
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def keyword_overlap_score(self, resume_keywords: List[str], 
                             job_keywords: List[str]) -> Dict[str, float]:
        """
        Calculate keyword overlap metrics
        
        Args:
            resume_keywords: Keywords from resume
            job_keywords: Keywords from job description
            
        Returns:
            Dictionary with various overlap metrics
        """
        resume_set = set(resume_keywords)
        job_set = set(job_keywords)
        
        matched_keywords = resume_set.intersection(job_set)
        
        # Calculate different metrics
        jaccard = self.jaccard_similarity(resume_set, job_set)
        
        # Percentage of job keywords found in resume
        match_rate = len(matched_keywords) / len(job_set) if job_set else 0.0
        
        return {
            'jaccard_similarity': round(jaccard, 4),
            'match_rate': round(match_rate, 4),
            'matched_keywords': list(matched_keywords),
            'matched_count': len(matched_keywords),
            'total_job_keywords': len(job_set),
            'coverage_percentage': round(match_rate * 100, 2)
        }
    
    def cosine_similarity_score(self, resume_text: str, job_text: str) -> float:
        """
        Calculate cosine similarity between resume and job description texts
        
        Args:
            resume_text: Resume text
            job_text: Job description text
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        try:
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
            
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return round(float(similarity), 4)
        except:
            return 0.0
    
    def calculate_weighted_score(self, resume_text: str, job_text: str,
                                resume_keywords: Dict[str, List[str]],
                                job_keywords: Dict[str, List[str]]) -> Dict[str, any]:
        """
        Calculate comprehensive weighted similarity score
        
        Args:
            resume_text: Full resume text
            job_text: Full job description text
            resume_keywords: Extracted resume keywords
            job_keywords: Extracted job description keywords
            
        Returns:
            Dictionary with detailed scoring information
        """
        # Calculate cosine similarity on full text
        text_similarity = self.cosine_similarity_score(resume_text, job_text)
        
        # Calculate keyword overlap for all keywords
        all_kw_overlap = self.keyword_overlap_score(
            resume_keywords.get('all_keywords', []),
            job_keywords.get('all_keywords', [])
        )
        
        # Calculate technical skills match (heavily weighted)
        skills_overlap = self.keyword_overlap_score(
            resume_keywords.get('technical_skills', []),
            job_keywords.get('technical_skills', [])
        )
        
        # Calculate TF-IDF keywords overlap
        tfidf_overlap = self.keyword_overlap_score(
            resume_keywords.get('tfidf_keywords', []),
            job_keywords.get('tfidf_keywords', [])
        )
        
        # Weighted scoring (optimized for skills-based matching)
        # If TF-IDF matching fails (returns 0 keywords), redistribute its weight to skills
        has_tfidf = len(resume_keywords.get('tfidf_keywords', [])) > 0 and len(job_keywords.get('tfidf_keywords', [])) > 0
        
        if has_tfidf:
            # Normal weights: Skills 50%, TF-IDF 25%, All keywords 15%, Text 10%
            weighted_score = (
                skills_overlap['match_rate'] * 0.50 +
                tfidf_overlap['match_rate'] * 0.25 +
                all_kw_overlap['match_rate'] * 0.15 +
                text_similarity * 0.10
            )
        else:
            # TF-IDF failed, redistribute: Skills 60%, All keywords 30%, Text 10%
            weighted_score = (
                skills_overlap['match_rate'] * 0.60 +
                all_kw_overlap['match_rate'] * 0.30 +
                text_similarity * 0.10
            )
        
        # Calculate overall percentage
        overall_percentage = round(weighted_score * 100, 2)
        
        # Determine match level (adjusted for skills-focused matching)
        if overall_percentage >= 80:
            match_level = "Excellent Match"
        elif overall_percentage >= 65:
            match_level = "Good Match"
        elif overall_percentage >= 50:
            match_level = "Moderate Match"
        elif overall_percentage >= 35:
            match_level = "Low Match"
        else:
            match_level = "Poor Match"
        
        return {
            'overall_score': round(weighted_score, 4),
            'overall_percentage': overall_percentage,
            'match_level': match_level,
            'detailed_scores': {
                'text_similarity': text_similarity,
                'skills_match_rate': skills_overlap['match_rate'],
                'tfidf_match_rate': tfidf_overlap['match_rate'],
                'all_keywords_match_rate': all_kw_overlap['match_rate']
            },
            'matched_skills': skills_overlap['matched_keywords'],
            'matched_tfidf_keywords': tfidf_overlap['matched_keywords'],
            'skills_coverage': skills_overlap['coverage_percentage'],
            'missing_skills': list(
                set(job_keywords.get('technical_skills', [])) - 
                set(resume_keywords.get('technical_skills', []))
            )
        }
    
    def generate_recommendations(self, score_results: Dict) -> List[str]:
        """
        Generate recommendations based on similarity score
        
        Args:
            score_results: Results from calculate_weighted_score
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        overall_pct = score_results['overall_percentage']
        missing_skills = score_results.get('missing_skills', [])
        
        if overall_pct < 50:
            recommendations.append("⚠️ Low match score. Consider tailoring your resume to this job description.")
        
        if missing_skills:
            recommendations.append(f"📝 Add these missing skills if you have them: {', '.join(missing_skills[:5])}")
        
        if score_results['detailed_scores']['skills_match_rate'] < 0.5:
            recommendations.append("🔧 Highlight more relevant technical skills from the job description.")
        
        if overall_pct >= 70:
            recommendations.append("✅ Strong match! Your resume aligns well with the job requirements.")
        
        return recommendations
