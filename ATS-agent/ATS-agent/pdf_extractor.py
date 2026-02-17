"""
PDF Text Extraction Module
Extracts text content from PDF files (resumes)
"""

import PyPDF2
import re
from typing import Optional, Dict, List
from pathlib import Path


class PDFExtractor:
    """Extract text from PDF files with format detection"""
    
    def __init__(self):
        # Common CV section headers
        self.cv_sections = [
            'experience', 'work experience', 'professional experience', 'employment',
            'education', 'academic background', 'qualifications',
            'skills', 'technical skills', 'core competencies', 'expertise',
            'projects', 'personal projects',
            'certifications', 'certificates', 'licenses',
            'summary', 'profile', 'objective', 'about me',
            'achievements', 'accomplishments', 'awards',
            'publications', 'research',
            'volunteer', 'volunteering',
            'languages', 'language proficiency', 'language skills',
            'interests', 'hobbies'
        ]
        
        # Common languages for detection
        self.common_languages = [
            'english', 'spanish', 'french', 'german', 'italian', 'portuguese',
            'mandarin', 'chinese', 'japanese', 'korean', 'arabic', 'russian',
            'hindi', 'bengali', 'punjabi', 'turkish', 'vietnamese', 'polish',
            'ukrainian', 'thai', 'dutch', 'greek', 'swedish', 'danish', 'norwegian',
            'finnish', 'czech', 'hungarian', 'romanian', 'indonesian', 'malay',
            'tagalog', 'swahili', 'hebrew', 'persian', 'urdu', 'tamil', 'telugu'
        ]
        
        # Proficiency levels
        self.proficiency_levels = [
            'native', 'fluent', 'proficient', 'advanced', 'intermediate', 'basic',
            'beginner', 'elementary', 'conversational', 'mother tongue', 'bilingual',
            'c2', 'c1', 'b2', 'b1', 'a2', 'a1'  # CEFR levels
        ]
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            Exception: If there's an error reading the PDF
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text from all pages
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                
                return text.strip()
                
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def extract_text_safe(self, pdf_path: str) -> Optional[str]:
        """
        Safely extract text from a PDF file, returning None on error
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string or None if extraction fails
        """
        try:
            return self.extract_text(pdf_path)
        except Exception as e:
            print(f"Warning: {str(e)}")
            return None
    
    def detect_file_format(self, file_path: str) -> Dict[str, any]:
        """
        Detect file format and properties
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with format information
        """
        path = Path(file_path)
        
        format_info = {
            'file_name': path.name,
            'file_extension': path.suffix.lower(),
            'file_size_kb': path.stat().st_size / 1024 if path.exists() else 0,
            'is_pdf': path.suffix.lower() == '.pdf',
            'is_docx': path.suffix.lower() in ['.docx', '.doc'],
            'is_txt': path.suffix.lower() == '.txt'
        }
        
        return format_info
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract content from each CV section
        
        Args:
            text: Extracted text from CV
            
        Returns:
            Dictionary mapping section names to their content
        """
        lines = text.split('\n')
        text_lower = text.lower()
        
        # Find section positions
        section_positions = []
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            for section in self.cv_sections:
                # Check if line is a section header
                if re.match(r'^' + re.escape(section) + r'\s*[:|\-|–]?\s*$', line_lower) or \
                   (section in line_lower and len(line_lower) < 50 and 
                    not any(word in line_lower for word in ['at', 'in', 'for', 'with', 'as'])):
                    section_positions.append({
                        'name': section,
                        'line': i,
                        'original_text': line.strip()
                    })
                    break
        
        # Sort by line number
        section_positions.sort(key=lambda x: x['line'])
        
        # Extract content between sections
        extracted_sections = {}
        for i, section_info in enumerate(section_positions):
            start_line = section_info['line'] + 1
            
            # Find end line (start of next section or end of document)
            if i < len(section_positions) - 1:
                end_line = section_positions[i + 1]['line']
            else:
                end_line = len(lines)
            
            # Extract content
            content_lines = lines[start_line:end_line]
            content = '\n'.join(line for line in content_lines if line.strip())
            
            section_name = section_info['name'].title()
            extracted_sections[section_name] = content.strip()
        
        return extracted_sections
    
    def extract_languages(self, text: str) -> Dict[str, any]:
        """
        Extract language information from CV text
        
        Args:
            text: Extracted text from CV
            
        Returns:
            Dictionary with language information
        """
        text_lower = text.lower()
        
        # First, try to get from dedicated Languages section
        sections = self.extract_sections(text)
        languages_section = None
        
        for section_name, content in sections.items():
            if 'language' in section_name.lower():
                languages_section = content
                break
        
        # Detect languages mentioned
        detected_languages = []
        
        # Search in languages section first, then entire text
        search_text = languages_section if languages_section else text_lower
        
        for language in self.common_languages:
            # Look for language name with word boundaries
            pattern = r'\b' + re.escape(language) + r'\b'
            if re.search(pattern, search_text):
                # Try to find proficiency level near the language
                language_info = {'language': language.title(), 'proficiency': None}
                
                # Search for proficiency within 50 characters of language mention
                lang_pos = search_text.find(language)
                if lang_pos != -1:
                    context = search_text[max(0, lang_pos-20):min(len(search_text), lang_pos+50)]
                    
                    for level in self.proficiency_levels:
                        if level in context:
                            language_info['proficiency'] = level.title()
                            break
                
                detected_languages.append(language_info)
        
        # Remove duplicates
        unique_languages = []
        seen = set()
        for lang in detected_languages:
            if lang['language'] not in seen:
                unique_languages.append(lang)
                seen.add(lang['language'])
        
        return {
            'languages_section': languages_section if languages_section else None,
            'detected_languages': unique_languages,
            'language_count': len(unique_languages),
            'has_language_section': languages_section is not None
        }
    
    def detect_cv_structure(self, text: str) -> Dict[str, any]:
        """
        Detect CV structure and format quality
        
        Args:
            text: Extracted text from CV
            
        Returns:
            Dictionary with structure information
        """
        text_lower = text.lower()
        lines = text.split('\n')
        
        # Detect sections present
        detected_sections = []
        for section in self.cv_sections:
            # Look for section headers (usually standalone or with colons/dashes)
            pattern = r'\b' + re.escape(section) + r'\b[\s:]*'
            if re.search(pattern, text_lower):
                detected_sections.append(section)
        
        # Extract section content
        extracted_content = self.extract_sections(text)
        
        # Extract languages
        language_info = self.extract_languages(text)
        
        # Detect contact information
        has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        has_phone = bool(re.search(r'[\+\(]?\d{1,4}[\)\-\s]?\(?\d{1,4}\)?[\-\s]?\d{1,4}[\-\s]?\d{1,9}', text))
        has_linkedin = 'linkedin' in text_lower
        has_github = 'github' in text_lower
        
        # Detect formatting quality
        non_empty_lines = [line for line in lines if line.strip()]
        avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0
        
        # Detect bullet points
        bullet_patterns = ['•', '●', '◦', '▪', '▫', '■', '□', '-', '*', '→']
        has_bullets = any(bullet in text for bullet in bullet_patterns)
        
        # Count dates (years in format YYYY or MM/YYYY)
        date_pattern = r'\b(19|20)\d{2}\b'
        dates_found = len(re.findall(date_pattern, text))
        
        # Detect formatting issues
        has_excessive_spaces = '    ' in text  # 4+ spaces
        has_strange_characters = bool(re.search(r'[^\w\s\-.,;:()@/•●◦▪▫■□*→\'"!?&#+=%]', text))
        
        # Calculate readability score
        readability_issues = []
        if avg_line_length < 10:
            readability_issues.append("Very short lines (possible extraction issue)")
        if not has_bullets:
            readability_issues.append("No bullet points detected")
        if len(text) < 500:
            readability_issues.append("Very short resume (less than 500 characters)")
        if has_strange_characters:
            readability_issues.append("Unusual characters detected (possible encoding issue)")
        
        # Categorize CV format
        cv_format = "Unknown"
        if len(detected_sections) >= 3:
            cv_format = "Well-Structured"
        elif len(detected_sections) >= 1:
            cv_format = "Basic Structure"
        else:
            cv_format = "Unstructured/Poor Format"
        
        # ATS-friendliness score
        ats_score = 0
        if len(detected_sections) >= 3: ats_score += 30
        if has_email: ats_score += 10
        if has_phone: ats_score += 10
        if has_bullets: ats_score += 15
        if dates_found >= 2: ats_score += 15
        if not has_strange_characters: ats_score += 10
        if avg_line_length > 20 and avg_line_length < 100: ats_score += 10
        
        return {
            'cv_format': cv_format,
            'detected_sections': detected_sections,
            'section_count': len(detected_sections),
            'extracted_sections': extracted_content,
            'languages': language_info,
            'has_contact_info': {
                'email': has_email,
                'phone': has_phone,
                'linkedin': has_linkedin,
                'github': has_github
            },
            'formatting': {
                'has_bullets': has_bullets,
                'dates_found': dates_found,
                'avg_line_length': round(avg_line_length, 2),
                'total_lines': len(lines),
                'non_empty_lines': len(non_empty_lines)
            },
            'quality_issues': readability_issues,
            'ats_friendly_score': ats_score,
            'ats_friendly_rating': self._get_ats_rating(ats_score)
        }
    
    def _get_ats_rating(self, score: int) -> str:
        """Get ATS-friendly rating based on score"""
        if score >= 80:
            return "Excellent - ATS Optimized"
        elif score >= 60:
            return "Good - ATS Friendly"
        elif score >= 40:
            return "Fair - Needs Improvement"
        else:
            return "Poor - Not ATS Friendly"
    
    def analyze_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Complete PDF analysis including format detection and text extraction
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with complete analysis
        """
        # Detect file format
        format_info = self.detect_file_format(pdf_path)
        
        # Extract text
        try:
            text = self.extract_text(pdf_path)
            
            # Get PDF metadata
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                metadata = {
                    'page_count': page_count,
                    'is_encrypted': pdf_reader.is_encrypted
                }
            
            # Detect structure
            structure_info = self.detect_cv_structure(text)
            
            return {
                'success': True,
                'file_info': format_info,
                'pdf_metadata': metadata,
                'structure_analysis': structure_info,
                'text_length': len(text),
                'extraction_quality': 'Good' if len(text) > 500 else 'Poor - Very short extraction'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'file_info': format_info
            }
