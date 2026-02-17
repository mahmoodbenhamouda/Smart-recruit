"""
LLM-based Resume Information Extractor
Extracts technical skills, soft skills, years of experience, and certifications using LLM
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path


class LLMResumeExtractor:
    """Extract structured information from resumes using LLM"""
    
    def __init__(self, provider: str = "openai", model: str = None, api_key: str = None):
        """
        Initialize LLM extractor
        
        Args:
            provider: "openai", "anthropic", "gemini", or "local"
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo", "claude-3", "gemini-pro")
            api_key: API key for the provider (or None to use environment variable)
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.client = None
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM client based on provider"""
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "local":
            self._init_local()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            import os
            
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            
            self.client = OpenAI(api_key=api_key)
            self.model = self.model or "gpt-3.5-turbo"  # Default model
            print(f"Initialized OpenAI with model: {self.model}")
            
        except ImportError:
            print("ERROR: openai package not installed. Install with: pip install openai")
            raise
    
    def _init_anthropic(self):
        """Initialize Anthropic (Claude) client"""
        try:
            import anthropic
            import os
            
            api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not provided.")
            
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = self.model or "claude-3-sonnet-20240229"
            print(f"Initialized Anthropic with model: {self.model}")
            
        except ImportError:
            print("ERROR: anthropic package not installed. Install with: pip install anthropic")
            raise
    
    def _init_gemini(self):
        """Initialize Google Gemini client"""
        try:
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            import os
            
            api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key not provided.")
            
            genai.configure(api_key=api_key)
            self.model = self.model or "gemini-2.5-flash"
            
            # Configure safety settings to be more permissive for resume content
            # Using BLOCK_ONLY_HIGH allows most content through
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            self.client = genai.GenerativeModel(
                self.model,
                safety_settings=self.safety_settings
            )
            print(f"Initialized Google Gemini with model: {self.model}")
            
        except ImportError:
            print("ERROR: google-generativeai package not installed. Install with: pip install google-generativeai")
            raise
    
    def _init_local(self):
        """Initialize local model (using transformers/ollama)"""
        print("⚠ Local model support - implement based on your preference (ollama, transformers, etc.)")
        # Placeholder for local model implementation
        raise NotImplementedError("Local model support not yet implemented. Use 'openai', 'anthropic', or 'gemini'.")
    
    def _create_extraction_prompt(self, resume_text: str) -> str:
        """Create a detailed prompt for structured extraction"""
        prompt = f"""Analyze the following resume and extract information in a structured JSON format.

Resume Text:
{resume_text}

Extract the following information:

1. **Technical Skills**: List all technical skills, tools, programming languages, frameworks, platforms, etc.
2. **Soft Skills**: List all soft skills like communication, leadership, teamwork, problem-solving, etc.
3. **Years of Experience**: 
   - Total years of professional experience
   - Years of experience per technical skill/domain (if mentioned)
4. **Certifications**: List all certifications, licenses, or professional credentials mentioned
5. **Education**: Degrees, institutions, graduation years
6. **Job Titles**: All job titles held
7. **Industries**: Industries worked in

Return ONLY a valid JSON object with this exact structure (no markdown, no code blocks):

{{
  "technical_skills": [
    {{"skill": "skill name", "years_experience": number or null, "proficiency": "beginner/intermediate/expert/null"}}
  ],
  "soft_skills": [
    {{"skill": "skill name", "context": "where mentioned or demonstrated"}}
  ],
  "total_experience_years": number,
  "certifications": [
    {{"name": "certification name", "issuer": "issuing organization", "year": number or null}}
  ],
  "education": [
    {{"degree": "degree name", "institution": "school name", "year": number or null, "field": "field of study"}}
  ],
  "job_titles": ["title1", "title2"],
  "industries": ["industry1", "industry2"],
  "summary": "brief professional summary"
}}

Be thorough and extract as much detail as possible. If information is not available, use null or empty array."""

        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the prompt"""
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "gemini":
            return self._call_gemini(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert resume analyzer. Extract information accurately and return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API"""
        response = self.client.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                # Don't set max_output_tokens=2000, it triggers safety blocks
                # Let the model use its default token limit
            }
        )
        
        # Check if response was blocked or has no content
        if not response.candidates:
            raise ValueError("No response candidates returned. Content may have been blocked.")
        
        candidate = response.candidates[0]
        
        # Check finish reason
        if candidate.finish_reason == 2:  # SAFETY
            # Print safety ratings for debugging
            print("\n⚠️ Response blocked by safety filters.")
            if hasattr(response, 'prompt_feedback'):
                print(f"Prompt feedback: {response.prompt_feedback}")
            if hasattr(candidate, 'safety_ratings'):
                print(f"Safety ratings: {candidate.safety_ratings}")
            raise ValueError("Response blocked by safety filters. The resume content may have triggered content safety policies.")
        elif candidate.finish_reason == 3:  # RECITATION
            raise ValueError("Response blocked due to recitation detection.")
        elif candidate.finish_reason not in [0, 1]:  # 0=UNSPECIFIED, 1=STOP (normal)
            raise ValueError(f"Unexpected finish reason: {candidate.finish_reason}")
        
        # Extract text from response
        if not candidate.content or not candidate.content.parts:
            raise ValueError("No content in response.")
        
        return response.text
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response from LLM, handling common formatting issues"""
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response:\n{response}")
            raise
    
    def extract_from_text(self, resume_text: str) -> Dict[str, Any]:
        """
        Extract structured information from resume text
        
        Args:
            resume_text: Resume text content
            
        Returns:
            Dictionary with extracted information
        """
        print(f"\n{'='*80}")
        print("EXTRACTING RESUME INFORMATION WITH LLM")
        print(f"{'='*80}\n")
        
        print(f"📄 Resume length: {len(resume_text)} characters")
        print(f"🤖 Using {self.provider.upper()} - {self.model}")
        print("\n⏳ Calling LLM for extraction...")
        
        # Create prompt and call LLM
        prompt = self._create_extraction_prompt(resume_text)
        response = self._call_llm(prompt)
        
        print("Received response from LLM")
        print("📊 Parsing results...\n")
        
        # Parse JSON response
        extracted_data = self._parse_json_response(response)
        
        # Add metadata
        extracted_data['_metadata'] = {
            'provider': self.provider,
            'model': self.model,
            'resume_length': len(resume_text)
        }
        
        return extracted_data
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract information from PDF resume
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted information
        """
        from pdf_extractor import PDFExtractor
        
        pdf_extractor = PDFExtractor()
        resume_text = pdf_extractor.extract_text(pdf_path)
        
        return self.extract_from_text(resume_text)
    
    def print_summary(self, data: Dict[str, Any]):
        """Print a formatted summary of extracted information"""
        print(f"\n{'='*80}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*80}\n")
        
        # Professional Summary
        if data.get('summary'):
            print("📝 Professional Summary:")
            print(f"   {data['summary']}\n")
        
        # Experience
        if data.get('total_experience_years'):
            print(f"💼 Total Experience: {data['total_experience_years']} years\n")
        
        # Technical Skills
        tech_skills = data.get('technical_skills', [])
        if tech_skills:
            print(f"💻 Technical Skills ({len(tech_skills)}):")
            for skill_obj in tech_skills[:20]:  # Show first 20
                skill_name = skill_obj.get('skill', skill_obj) if isinstance(skill_obj, dict) else skill_obj
                years = skill_obj.get('years_experience') if isinstance(skill_obj, dict) else None
                proficiency = skill_obj.get('proficiency') if isinstance(skill_obj, dict) else None
                
                details = []
                if years:
                    details.append(f"{years} years")
                if proficiency:
                    details.append(proficiency)
                
                detail_str = f" ({', '.join(details)})" if details else ""
                print(f"   • {skill_name}{detail_str}")
            
            if len(tech_skills) > 20:
                print(f"   ... and {len(tech_skills) - 20} more")
            print()
        
        # Soft Skills
        soft_skills = data.get('soft_skills', [])
        if soft_skills:
            print(f"🤝 Soft Skills ({len(soft_skills)}):")
            for skill_obj in soft_skills[:15]:  # Show first 15
                skill_name = skill_obj.get('skill', skill_obj) if isinstance(skill_obj, dict) else skill_obj
                print(f"   • {skill_name}")
            
            if len(soft_skills) > 15:
                print(f"   ... and {len(soft_skills) - 15} more")
            print()
        
        # Certifications
        certifications = data.get('certifications', [])
        if certifications:
            print(f"🏆 Certifications ({len(certifications)}):")
            for cert in certifications:
                if isinstance(cert, dict):
                    name = cert.get('name', '')
                    issuer = cert.get('issuer', '')
                    year = cert.get('year', '')
                    cert_str = name
                    if issuer:
                        cert_str += f" - {issuer}"
                    if year:
                        cert_str += f" ({year})"
                    print(f"   • {cert_str}")
                else:
                    print(f"   • {cert}")
            print()
        
        # Education
        education = data.get('education', [])
        if education:
            print(f"🎓 Education ({len(education)}):")
            for edu in education:
                if isinstance(edu, dict):
                    degree = edu.get('degree', '')
                    institution = edu.get('institution', '')
                    year = edu.get('year', '')
                    field = edu.get('field', '')
                    
                    edu_str = degree
                    if field:
                        edu_str += f" in {field}"
                    if institution:
                        edu_str += f" - {institution}"
                    if year:
                        edu_str += f" ({year})"
                    print(f"   • {edu_str}")
                else:
                    print(f"   • {edu}")
            print()
        
        # Job Titles
        job_titles = data.get('job_titles', [])
        if job_titles:
            print(f"👔 Job Titles ({len(job_titles)}):")
            for title in job_titles:
                print(f"   • {title}")
            print()
        
        # Industries
        industries = data.get('industries', [])
        if industries:
            print(f"🏭 Industries:")
            for industry in industries:
                print(f"   • {industry}")
            print()
    
    def save_results(self, data: Dict[str, Any], output_path: str = "llm_extraction_results.json"):
        """Save extraction results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")


def main():
    """Example usage"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='LLM-based Resume Information Extractor')
    parser.add_argument('resume', help='Path to resume PDF file')
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'gemini'], 
                       default='openai', help='LLM provider to use')
    parser.add_argument('--model', help='Model name (e.g., gpt-4, gpt-3.5-turbo)')
    parser.add_argument('--api-key', help='API key (or set via environment variable)')
    parser.add_argument('--output', default='llm_extraction_results.json',
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Check if API key is available
    env_var_map = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'gemini': 'GOOGLE_API_KEY'
    }
    
    env_var = env_var_map[args.provider]
    if not args.api_key and not os.getenv(env_var):
        print(f"\n❌ ERROR: No API key found!")
        print(f"Either:")
        print(f"  1. Set environment variable: {env_var}")
        print(f"  2. Pass --api-key parameter")
        return
    
    try:
        # Initialize extractor
        extractor = LLMResumeExtractor(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key
        )
        
        # Extract information
        results = extractor.extract_from_pdf(args.resume)
        
        # Print summary
        extractor.print_summary(results)
        
        # Save results
        extractor.save_results(results, args.output)
        
        print(f"\n{'='*80}")
        print("✅ EXTRACTION COMPLETE!")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
