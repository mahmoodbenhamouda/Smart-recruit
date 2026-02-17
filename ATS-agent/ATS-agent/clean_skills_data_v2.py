"""
Enhanced Skills Dataset Cleaner v2
Intelligent filtering to remove non-skills, fix formatting, and validate skills
"""

import pandas as pd
import re
from collections import Counter
from typing import List, Set
import numpy as np


class EnhancedSkillsCleaner:
    """Clean and normalize skills dataset with intelligent filtering"""
    
    def __init__(self):
        # Common multi-word skills that should NOT be split
        self.multi_word_skills = {
            # Programming & Development
            'machine learning', 'deep learning', 'natural language processing', 'computer vision',
            'data science', 'data analysis', 'data analytics', 'data engineering', 'data mining',
            'software development', 'web development', 'mobile development', 'full stack',
            'front end', 'back end', 'frontend', 'backend', 'agile development', 
            'object oriented programming', 'test driven development', 'continuous integration', 
            'continuous deployment', 'version control', 'source control', 'code review',
            
            # Cloud & Infrastructure
            'cloud computing', 'amazon web services', 'microsoft azure', 'google cloud',
            'infrastructure as code', 'configuration management', 'containerization',
            'cloud architecture', 'cloud security', 'devops', 'site reliability',
            
            # Business & Management
            'project management', 'product management', 'business analysis', 'business intelligence',
            'business development', 'strategic planning', 'change management', 'risk management',
            'supply chain', 'supply chain management', 'inventory management', 'quality assurance',
            'quality control', 'process improvement', 'continuous improvement', 'performance management',
            'stakeholder management', 'vendor management', 'budget management', 'financial analysis',
            'financial modeling', 'cost analysis', 'root cause analysis', 'gap analysis',
            'account management', 'client management', 'relationship management',
            
            # Analytics & Statistics
            'statistical analysis', 'predictive modeling', 'predictive analytics', 'time series',
            'time series analysis', 'regression analysis', 'sentiment analysis', 'customer analytics',
            'marketing analytics', 'web analytics', 'social media analytics', 'business analytics',
            'data visualization', 'data warehousing', 'data governance',
            
            # Design & Creative
            'user experience', 'user interface', 'graphic design', 'web design', 'ui design',
            'ux design', 'visual design', 'interaction design', 'responsive design',
            'ui ux', 'ux ui', 'product design', 'service design',
            
            # Soft Skills
            'problem solving', 'critical thinking', 'analytical thinking', 'creative thinking',
            'decision making', 'time management', 'stress management', 'conflict resolution',
            'team building', 'public speaking', 'cross functional', 'customer service',
            'interpersonal skills', 'communication skills', 'leadership skills', 'teamwork',
            'attention to detail', 'organizational skills', 'multitasking',
            
            # Operations & Manufacturing
            'operations management', 'process optimization', 'lean manufacturing', 'six sigma',
            'total quality management', 'just in time', 'demand planning', 'capacity planning',
            'production planning', 'resource planning', 'workforce planning', 'lean six sigma',
            'kaizen', 'root cause', 'continuous improvement',
            
            # Frameworks & Tools
            'spring boot', 'ruby on rails', 'entity framework', 'apache spark', 'big data',
            'microsoft excel', 'microsoft office', 'google analytics', 'google sheets',
            'power bi', 'sql server', 'visual studio', 'react native', 'node js',
            
            # Methodologies
            'agile methodology', 'scrum methodology', 'waterfall methodology', 'kanban',
            'design thinking', 'human centered design', 'agile scrum',
            
            # Healthcare & Safety
            'patient care', 'emergency response', 'first aid', 'cpr', 'basic life support',
            'advanced life support', 'infection control', 'medical terminology',
            'electronic medical records', 'hipaa compliance',
            
            # Education & Certifications
            'high school diploma', 'bachelor degree', 'master degree', 'phd', 'mba',
            'associate degree', 'professional development', 'continuing education',
            
            # Sales & Marketing
            'sales management', 'digital marketing', 'content marketing', 'email marketing',
            'social media', 'social media marketing', 'search engine optimization', 'seo',
            'search engine marketing', 'sem', 'pay per click', 'ppc', 'lead generation',
            'customer relationship management', 'crm',
            
            # Finance & Accounting
            'financial reporting', 'financial planning', 'accounts payable', 'accounts receivable',
            'general ledger', 'cost accounting', 'tax preparation', 'financial statements',
            'balance sheet', 'income statement', 'cash flow',
            
            # HR & Recruitment
            'human resources', 'talent acquisition', 'employee relations', 'performance reviews',
            'talent management', 'workforce management', 'compensation benefits',
            'onboarding', 'talent development',
            
            # Legal & Compliance
            'regulatory compliance', 'contract management', 'legal research', 'contract negotiation',
            'intellectual property', 'employment law', 'corporate law',
            
            # Construction & Trades
            'construction management', 'building codes', 'safety regulations', 'osha',
            'construction safety', 'project scheduling', 'blueprint reading',
            
            # Hospitality & Service
            'food safety', 'food service', 'customer satisfaction', 'guest services',
            'restaurant management', 'hotel management', 'event planning',
            
            # Logistics & Transportation
            'logistics management', 'transportation management', 'warehouse management',
            'route planning', 'fleet management', 'distribution management',
        }
        
        # Phrases that indicate NON-skill entries (job duties, requirements, etc.)
        self.non_skill_phrases = {
            'experience in', 'experience with', 'knowledge of', 'ability to',
            'responsible for', 'duties include', 'years of', 'working with',
            'familiarity with', 'understanding of', 'exposure to', 'comfortable with',
            'skilled in', 'proficient in', 'expert in', 'background in',
            'must have', 'should have', 'required to', 'able to', 'willing to',
            'demonstrated ability', 'proven ability', 'strong knowledge',
            'minimum of', 'at least', 'or more', 'or equivalent',
            'post construction', 'site visits', 'compliance site', 'ground water',
            'level iv', 'level iii', 'level ii', 'level i', 'grade',
            'extra hours', 'working hours', 'shift work', 'overtime',
            'repetitive', 'physically demanding', 'heavy lifting',
            'transfer station', 'pumpage inventories', 'tenure and promotion',
            'directory maintenance', 'recovery processes', 'testing service',
        }
        
        # Generic/vague terms that aren't specific skills
        self.vague_terms = {
            'skills', 'knowledge', 'abilities', 'competencies', 'proficiencies',
            'expertise', 'capabilities', 'qualifications', 'requirements',
            'responsibilities', 'duties', 'tasks', 'activities', 'functions',
            'operations', 'procedures', 'processes', 'methods', 'techniques',
            'tools', 'systems', 'applications', 'platforms', 'technologies',
            'programs', 'services', 'products', 'solutions', 'resources',
            'materials', 'equipment', 'facilities', 'environment', 'setting',
            'area', 'field', 'domain', 'industry', 'sector', 'market',
            'business', 'company', 'organization', 'department', 'team',
            'position', 'role', 'job', 'work', 'career', 'professional',
            'certification', 'license', 'credential', 'diploma', 'certificate',
            'training', 'education', 'degree', 'course', 'program', 'class',
            'experience', 'optional', 'desirable', 'desired', 'preferred', 'required',
            'if applicable', 'one', 'two', 'three', 'four', 'five',
            'physical requirements', 'physical demands', 'physical ability', 'physical abilities',
            'certifications', 'licensure', 'registration', 'software', 'system', 'standards',
            'travel', 'lifting', 'management', 'compliance', 'applicable',
            # Additional vague terms from data
            'asset', 'benefits', 'pto', 'advantageous', 'plan', 'process',
            'designation', 'bonus', 'clearance', 'if required', 'regulations',
            'testing', 'design', 'analysis', 'documentation', 'guidelines',
            'membership', 'programming', 'engineering', 'nursing', 'process',
        }
        
        # Common noise words
        self.noise_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'a', 'an', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
            'can', 'must', 'shall', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'etc',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'this', 'that',
            'including', 'required', 'preferred', 'plus', 'strong', 'excellent',
            'good', 'proficient', 'familiar', 'working', 'demonstrated',
        }
        
        # Separators for splitting skills
        self.separators = [
            ',', ';', '|', '/', '\\', '-', '–', '—',
            '\n', '\t', '•', '·', '*', '>', '<',
            '(', ')', '[', ']', '{', '}'
        ]
        
        # Skill normalizations (expand abbreviations, fix common issues)
        self.normalizations = {
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            'dl': 'deep learning',
            'aws': 'amazon web services',
            'gcp': 'google cloud platform',
            'sql': 'sql',
            'nosql': 'nosql',
            'js': 'javascript',
            'ts': 'typescript',
            'py': 'python',
            'c++': 'c plus plus',
            'c#': 'c sharp',
            'devops': 'devops',
            'ci/cd': 'continuous integration continuous deployment',
            'cicd': 'continuous integration continuous deployment',
            'k8s': 'kubernetes',
            'crm': 'customer relationship management',
            'erp': 'enterprise resource planning',
            'api': 'api',
            'rest': 'rest api',
            'html': 'html',
            'css': 'css',
            'ui/ux': 'user interface user experience',
            'b2b': 'business to business',
            'b2c': 'business to consumer',
            'kpi': 'key performance indicators',
            'roi': 'return on investment',
            'seo': 'search engine optimization',
            'sem': 'search engine marketing',
            'ppc': 'pay per click',
            'rn': 'registered nurse',
            'lpn': 'licensed practical nurse',
            'cna': 'certified nursing assistant',
            'cpr': 'cardiopulmonary resuscitation',
            'bls': 'basic life support',
            'acls': 'advanced cardiovascular life support',
            'pmp': 'project management professional',
            'scrum': 'scrum',
            'agile': 'agile',
            'lean': 'lean',
            'six sigma': 'six sigma',
            'iso': 'iso',
            'osha': 'occupational safety and health administration',
            'hipaa': 'health insurance portability and accountability act',
            'gdpr': 'general data protection regulation',
        }
        
    def is_valid_skill(self, skill: str) -> bool:
        """
        Check if a string is a valid skill
        
        Args:
            skill: Skill text to validate
            
        Returns:
            True if valid skill, False otherwise
        """
        skill_lower = skill.lower().strip()
        
        # Too short or too long
        if len(skill_lower) < 2 or len(skill_lower) > 60:
            return False
        
        # Just a number or date
        if skill_lower.isdigit() or re.match(r'^\d{2,4}$', skill_lower):
            return False
        
        # Contains year patterns (2020, 2021, etc.)
        if re.search(r'\b(19|20)\d{2}\b', skill_lower):
            return False
        
        # Contains date patterns
        if re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', skill_lower):
            return False
        
        # Check for non-skill phrases
        for phrase in self.non_skill_phrases:
            if phrase in skill_lower:
                return False
        
        # Just a vague term on its own
        if skill_lower in self.vague_terms:
            return False
        
        # Just a noise word
        if skill_lower in self.noise_words:
            return False
        
        # Too many numbers (likely a code or ID)
        if sum(c.isdigit() for c in skill_lower) > len(skill_lower) * 0.3:
            return False
        
        # Contains special characters that indicate noise
        if re.search(r'[#@$%^&*<>{}[\]\\]', skill_lower):
            return False
        
        # Looks like an email or URL
        if '@' in skill_lower or 'www.' in skill_lower or 'http' in skill_lower:
            return False
        
        # Contains "experience" followed by number (e.g., "experience 5 years")
        if re.search(r'experience\s+\d+', skill_lower):
            return False
        
        # Starts with number followed by "years" or "year"
        if re.match(r'^\d+\s*(years?|yrs?|months?)', skill_lower):
            return False
        
        # Contains "+ years" pattern (e.g., "2+ years", "3+ years")
        if re.search(r'\d+\+?\s*(years?|yrs?)', skill_lower):
            return False
        
        # Contains level indicators (level i, level ii, etc.)
        if re.search(r'level\s+(i{1,3}|iv|v|1|2|3|4|5)', skill_lower):
            return False
        
        # Too many words (likely a sentence or description)
        word_count = len(skill_lower.split())
        if word_count > 8:
            return False
        
        # Single or two-letter abbreviations are too ambiguous (except known ones)
        known_short_abbrevs = {
            'r', 'c', 'ai', 'ml', 'bi', 'qa', 'ui', 'ux', 'hr', 'it', 'js', 'py',
            'sql', 'api', 'aws', 'gcp', 'sap', 'erp', 'crm', 'ehr', 'emr', 'pos',
            'cad', 'cpr', 'bls', 'acls', 'pals', 'nrp', 'aha', 'arrt', 'ascp',
            'cpa', 'cma', 'rn', 'lpn', 'cna', 'bsn', 'msn', 'mba', 'phd', 'pe',
            'pmp', 'cdl', 'ged', 'fsa', 'cms', 'dod', 'ppe', 'osha', 'hipaa'
        }
        if len(skill_lower) <= 2 and skill_lower not in known_short_abbrevs:
            return False
        
        # Contains "or" or "and" connecting unrelated things
        if word_count > 4 and (' or ' in skill_lower or ' and ' in skill_lower):
            # Exception: known multi-word skills
            if skill_lower not in self.multi_word_skills:
                return False
        
        return True
    
    def fix_spacing(self, skill: str) -> str:
        """
        Fix common spacing issues in skills
        
        Args:
            skill: Skill text
            
        Returns:
            Fixed skill text
        """
        # Common concatenated words
        spacing_fixes = {
            'problemsolving': 'problem solving',
            'decisionmaking': 'decision making',
            'timemanagement': 'time management',
            'projectmanagement': 'project management',
            'customerservice': 'customer service',
            'machinelearning': 'machine learning',
            'deeplearning': 'deep learning',
            'dataanalysis': 'data analysis',
            'datascience': 'data science',
            'softwareengineer': 'software engineer',
            'webdevelopment': 'web development',
            'frontenddevelopment': 'frontend development',
            'backenddevelopment': 'backend development',
            'fullstackdevelopment': 'full stack development',
        }
        
        skill_lower = skill.lower()
        for concat, fixed in spacing_fixes.items():
            if concat in skill_lower:
                skill = skill.replace(concat, fixed).replace(concat.title(), fixed.title())
        
        # Fix double apostrophes (bachelor''s -> bachelor's)
        skill = skill.replace("''", "'")
        
        # Fix apostrophes
        skill = re.sub(r'\s+s\s+', "'s ", skill)  # "bachelor s degree" -> "bachelor's degree"
        skill = re.sub(r'\bs\s+(degree|diploma)', r"'s \1", skill)
        
        # Fix multiple spaces
        skill = re.sub(r'\s+', ' ', skill)
        
        return skill.strip()
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def split_skills(self, text: str) -> List[str]:
        """
        Split text into individual skills while preserving multi-word skills
        
        Args:
            text: Text containing multiple skills
            
        Returns:
            List of individual skills
        """
        if not text:
            return []
        
        # First, protect multi-word skills by replacing spaces with placeholders
        protected_text = text
        placeholder_map = {}
        
        for i, multi_word in enumerate(sorted(self.multi_word_skills, key=len, reverse=True)):
            if multi_word in protected_text:
                placeholder = f"__MULTIWORD_{i}__"
                placeholder_map[placeholder] = multi_word
                protected_text = protected_text.replace(multi_word, placeholder)
        
        # Now split by separators
        pattern = '|'.join(re.escape(sep) for sep in self.separators)
        skills = re.split(pattern, protected_text)
        
        # Restore multi-word skills
        restored_skills = []
        for skill in skills:
            for placeholder, original in placeholder_map.items():
                skill = skill.replace(placeholder, original)
            if skill.strip():
                restored_skills.append(skill.strip())
        
        return restored_skills
    
    def normalize_skill(self, skill: str) -> str:
        """
        Normalize skill name (expand abbreviations, standardize format)
        
        Args:
            skill: Skill text
            
        Returns:
            Normalized skill text
        """
        # Fix spacing issues FIRST
        skill = self.fix_spacing(skill)
        
        skill_lower = skill.lower().strip()
        
        # Apply normalizations
        if skill_lower in self.normalizations:
            return self.normalizations[skill_lower]
        
        return skill_lower
    
    def clean_dataframe(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Clean the entire dataframe
        
        Args:
            df: DataFrame with skills data
            column_name: Name of column containing skills
            
        Returns:
            Cleaned DataFrame
        """
        print(f"📊 Starting with {len(df)} rows")
        
        # Step 1: Clean text
        print("🧹 Cleaning text...")
        df['cleaned'] = df[column_name].apply(self.clean_text)
        
        # Step 2: Split skills
        print("✂️ Splitting skills while preserving multi-word skills...")
        df['skills_list'] = df['cleaned'].apply(self.split_skills)
        
        # Step 3: Explode into individual skills
        print("💥 Exploding into individual skills...")
        df_exploded = df.explode('skills_list')
        df_exploded = df_exploded[df_exploded['skills_list'].notna()]
        df_exploded = df_exploded[df_exploded['skills_list'] != '']
        
        print(f"   Before filtering: {len(df_exploded)} skills")
        
        # Step 4: Validate skills (remove non-skills)
        print("🔍 Filtering out non-skills (job duties, requirements, etc.)...")
        df_exploded = df_exploded[df_exploded['skills_list'].apply(self.is_valid_skill)]
        print(f"   After filtering: {len(df_exploded)} valid skills")
        
        # Step 5: Normalize skill names
        print("🔄 Normalizing skill names (expanding abbreviations, fixing spacing)...")
        df_exploded['skill'] = df_exploded['skills_list'].apply(self.normalize_skill)
        
        # Calculate frequencies BEFORE deduplication
        print("📊 Calculating skill frequencies...")
        skill_counts = df_exploded['skill'].value_counts()
        
        # Step 6: Remove duplicates
        print("🗑️ Removing duplicates...")
        print(f"   Before deduplication: {len(df_exploded)} skills")
        df_clean = df_exploded[['skill']].drop_duplicates()
        print(f"   After deduplication: {len(df_clean)} unique skills")
        print(f"   Removed {len(df_exploded) - len(df_clean)} duplicates")
        
        # Add frequency counts
        df_clean['frequency'] = df_clean['skill'].map(skill_counts)
        df_clean = df_clean.sort_values('frequency', ascending=False).reset_index(drop=True)
        
        return df_clean
    
    def filter_by_frequency(self, df: pd.DataFrame, min_freq: int = 2) -> pd.DataFrame:
        """
        Filter out very rare skills
        
        Args:
            df: DataFrame with skills
            min_freq: Minimum frequency to keep
            
        Returns:
            Filtered DataFrame
        """
        print(f"\n📊 Filtering skills by frequency (min: {min_freq})...")
        skill_counts = df['skill'].value_counts()
        valid_skills = skill_counts[skill_counts >= min_freq].index
        df_filtered = df[df['skill'].isin(valid_skills)]
        print(f"   Kept {len(df_filtered)} skills that appear at least {min_freq} times")
        print(f"   Removed {len(df) - len(df_filtered)} rare skills")
        return df_filtered


def main():
    """Main execution function"""
    print("=" * 80)
    print("🧹 ENHANCED SKILLS DATASET CLEANER V2")
    print("=" * 80)
    
    # Configuration
    input_file = 'data/skills_exploded (2).csv'
    output_file = 'data/skills_cleaned_v2.csv'
    stats_file = 'data/skills_statistics_v2.csv'
    
    # Load data
    print(f"\n📂 Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df)} rows")
    
    print(f"\n📋 Columns: {df.columns.tolist()}")
    
    # Determine which column contains skills
    skill_column = None
    for col in df.columns:
        if 'skill' in col.lower():
            skill_column = col
            break
    
    if skill_column is None:
        skill_column = df.columns[0]
        print(f"⚠️ No 'skill' column found, using first column: '{skill_column}'")
    else:
        print(f"🎯 Using column: '{skill_column}'")
    
    # Show sample
    print(f"\n📝 Sample raw data:")
    print(df[skill_column].head(3).tolist())
    
    # Clean the data
    print(f"\n{'=' * 80}")
    print("🚀 STARTING CLEANING PROCESS")
    print(f"{'=' * 80}\n")
    
    cleaner = EnhancedSkillsCleaner()
    df_clean = cleaner.clean_dataframe(df, skill_column)
    
    # Optional: Filter by frequency (uncomment to use)
    # df_clean = cleaner.filter_by_frequency(df_clean, min_freq=2)
    
    # Generate statistics
    print(f"\n{'=' * 80}")
    print("📊 STATISTICS")
    print(f"{'=' * 80}\n")
    
    print(f"✅ Total unique skills: {len(df_clean)}")
    print(f"✅ Total skill mentions: {df_clean['frequency'].sum():,}")
    
    print(f"\n🔝 Top 50 most common skills:")
    for i, row in df_clean.head(50).iterrows():
        print(f"   {i+1:2d}. {row['skill']:50s} ({row['frequency']:,} occurrences)")
    
    # Additional stats
    print(f"\n📈 Frequency distribution:")
    print(f"   • Skills appearing 100+ times: {len(df_clean[df_clean['frequency'] >= 100]):,}")
    print(f"   • Skills appearing 10-99 times: {len(df_clean[(df_clean['frequency'] >= 10) & (df_clean['frequency'] < 100)]):,}")
    print(f"   • Skills appearing 2-9 times: {len(df_clean[(df_clean['frequency'] >= 2) & (df_clean['frequency'] < 10)]):,}")
    print(f"   • Skills appearing only once: {len(df_clean[df_clean['frequency'] == 1]):,}")
    
    # Save results
    print(f"\n{'=' * 80}")
    print("💾 SAVING RESULTS")
    print(f"{'=' * 80}\n")
    
    # Fix double apostrophes before saving
    df_clean['skill'] = df_clean['skill'].str.replace("''", "'", regex=False)
    
    df_clean.to_csv(output_file, index=False)
    print(f"✅ Saved cleaned skills to: {output_file}")
    print(f"   (includes frequency counts)")
    
    print(f"\n{'=' * 80}")
    print("✅ CLEANING COMPLETE!")
    print(f"{'=' * 80}\n")
    
    print(f"📊 Summary:")
    print(f"   • Input: {len(df):,} rows")
    print(f"   • Output: {len(df_clean):,} unique clean skills")
    print(f"   • Reduction: {((len(df) - len(df_clean)) / len(df) * 100):.1f}%")


if __name__ == "__main__":
    main()
