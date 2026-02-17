"""
Keyword Extraction Module
Extracts important keywords from text using NLP techniques
"""

import re
from typing import List, Set, Dict
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


class KeywordExtractor:
    """Extract keywords from text using various NLP techniques"""
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize the keyword extractor
        
        Args:
            use_spacy: Whether to use spaCy for advanced NLP processing
        """
        # Store whether to use spaCy for NLP processing (default: True)
        self.use_spacy = use_spacy
        
        # Initialize spaCy NLP pipeline as None (will be loaded if use_spacy=True)
        self.nlp = None
        
        # Only load spaCy models if NLP processing is enabled
        if use_spacy:
            try:
                # Try to load the medium model first (40MB, 514K vocab, better for NER and entity recognition)
                # This model has better accuracy for detecting organizations, locations, skills
                self.nlp = spacy.load("en_core_web_md")
                print("[OK] Loaded spaCy model: en_core_web_md")
            except OSError:
                # If medium model not found, fall back to small model (12MB, 20K vocab)
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    print("[OK] Loaded spaCy model: en_core_web_sm")
                except OSError:
                    # If no spaCy model found, disable spaCy and print installation instructions
                    print("[WARNING] spaCy model not found. Install with: python -m spacy download en_core_web_md")
                    self.use_spacy = False
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned text
        """
        # Convert all text to lowercase for consistent matching
        # Example: "Python Developer" → "python developer"
        text = text.lower()
        
        # Remove special characters but keep alphanumeric, spaces, +, and #
        # This preserves: "C++", "C#" while removing quotes, brackets, etc.
        # Regex: [^...] = match anything NOT in this set
        # \w = word characters (a-z, A-Z, 0-9, _)
        # \s = whitespace
        text = re.sub(r'[^a-zA-Z0-9\s\+\#]', ' ', text)
        
        # Remove extra whitespace (multiple spaces → single space)
        # split() splits by any whitespace, join(' ') joins with single space
        # Example: "python    developer" → "python developer"
        text = ' '.join(text.split())
        
        return text
    
    def extract_keywords_spacy(self, text: str, top_n: int = 20) -> List[str]:
        """
        Extract keywords using spaCy NLP
        
        Args:
            text: Text to extract keywords from
            top_n: Number of top keywords to return
            
        Returns:
            List of extracted keywords
        """
        # If spaCy model is not loaded, return empty list
        if not self.nlp:
            return []
        
        # Process text through spaCy's NLP pipeline
        # This performs: tokenization, POS tagging, NER, dependency parsing
        doc = self.nlp(text)
        
        # Initialize list to store extracted keywords
        keywords = []
        
        # Step 1: Extract named entities using NER (Named Entity Recognition)
        # doc.ents contains all detected entities with their labels
        for ent in doc.ents:
            # Only keep entities that are relevant for resumes/job descriptions
            # PERSON: People's names (e.g., "John Smith")
            # ORG: Organizations/companies (e.g., "Google", "Microsoft")
            # PRODUCT: Products/technologies (e.g., "AWS", "iPhone")
            # SKILL: Skills (if detected by model)
            # GPE: Geopolitical entities/locations (e.g., "New York", "USA")
            if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'SKILL', 'GPE']:
                # Add entity text in lowercase for consistency
                keywords.append(ent.text.lower())
        
        # Step 2: Extract important parts of speech (nouns, proper nouns, adjectives)
        # Iterate through each token (word) in the document
        for token in doc:
            # token.pos_ = part of speech tag
            # NOUN: common nouns (e.g., "developer", "experience")
            # PROPN: proper nouns (e.g., "Python", "AWS")
            # ADJ: adjectives (e.g., "senior", "experienced")
            # Filter out stop words (common words like "the", "is", "a")
            # Filter out very short tokens (< 3 characters)
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop and len(token.text) > 2:
                # Use lemma (base form) instead of inflected form
                # Example: "developers" → "developer", "experienced" → "experience"
                keywords.append(token.lemma_.lower())
        
        # Step 3: Count frequency of each keyword
        # Counter creates a dictionary with keyword counts
        # Example: Counter(['python', 'aws', 'python', 'docker']) → {'python': 2, 'aws': 1, 'docker': 1}
        keyword_freq = Counter(keywords)
        
        # Step 4: Return top N most frequent keywords
        # most_common(top_n) returns list of (keyword, count) tuples sorted by frequency
        # List comprehension extracts only the keywords (ignoring counts)
        return [kw for kw, _ in keyword_freq.most_common(top_n)]
    
    def extract_keywords_tfidf(self, text: str, top_n: int = 20) -> List[str]:
        """
        Extract keywords using TF-IDF
        
        Args:
            text: Text to extract keywords from
            top_n: Number of top keywords to return
            
        Returns:
            List of extracted keywords
        """
        # Step 1: Clean and preprocess the text (lowercase, remove special chars)
        cleaned_text = self.preprocess_text(text)
        
        # Step 2: Define custom stop words to filter out
        # These are common but not meaningful words that should be excluded
        custom_stop_words = [
            'company', 'city', 'state', 'location', 'email', 'phone', 'address',
            'january', 'february', 'march', 'april', 'may', 'june', 'july',
            'august', 'september', 'october', 'november', 'december',
            'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
            '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'
        ]
        
        # Step 3: Initialize TF-IDF vectorizer with specific parameters
        # TF-IDF = Term Frequency - Inverse Document Frequency
        # Scores words based on: frequency in document vs frequency across documents
        vectorizer = TfidfVectorizer(
            max_features=top_n * 2,  # Get 2x features to have buffer after filtering
            stop_words='english',     # Remove common English stop words ("the", "is", "a")
            ngram_range=(1, 3),       # Extract 1-word, 2-word, and 3-word phrases
            min_df=1,                 # Minimum document frequency (keep all terms)
            max_df=0.95               # Maximum document frequency (remove very common terms)
        )
        
        try:
            # Step 4: Fit vectorizer and transform text into TF-IDF matrix
            # Result: sparse matrix of shape (1, num_features)
            tfidf_matrix = vectorizer.fit_transform([cleaned_text])
            
            # Step 5: Get feature names (the actual words/phrases)
            # Example: ["python", "machine learning", "aws", "data science", ...]
            feature_names = vectorizer.get_feature_names_out()
            
            # Step 6: Extract TF-IDF scores for each feature
            # Convert sparse matrix to dense array and get first (only) row
            # scores[i] = TF-IDF score for feature_names[i]
            scores = tfidf_matrix.toarray()[0]
            
            # Step 7: Create list of (keyword, score) tuples
            # zip() pairs each feature name with its score
            keyword_scores = list(zip(feature_names, scores))
            
            # Step 8: Sort by score in descending order (highest scores first)
            # lambda x: x[1] means sort by second element of tuple (the score)
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Step 9: Filter out custom stop words and very short keywords
            # Only keep keywords that:
            # - Are NOT in custom_stop_words list
            # - Have length > 2 characters
            filtered_keywords = [
                kw for kw, score in keyword_scores 
                if kw not in custom_stop_words and len(kw) > 2
            ]
            
            # Step 10: Return top N filtered keywords
            return filtered_keywords[:top_n]
        except:
            # If any error occurs (empty text, vectorization fails, etc.), return empty list
            return []
    
    def extract_technical_skills(self, text: str) -> Set[str]:
        """
        Extract common technical skills and programming languages
        
        Args:
            text: Text to extract skills from
            
        Returns:
            Set of detected skills
        """
        # Convert entire text to lowercase for case-insensitive matching
        # Example: "Python" and "python" will both match
        text_lower = text.lower()
        
        # Comprehensive technical skills and tools database (~200+ skills)
        # This is a hardcoded database of known technical terms
        skills_database = {
            # Programming languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c ', 'ruby', 'go', 'golang', 'rust',
            'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'perl', 'bash', 'powershell',
            'vba', 'sas', 'julia', 'dart', 'objective-c',
            
            # Web technologies
            'html', 'html5', 'css', 'css3', 'react', 'reactjs', 'angular', 'angularjs', 'vue', 'vuejs',
            'node.js', 'nodejs', 'django', 'flask', 'spring', 'spring boot', 'express', 'expressjs',
            'fastapi', 'next.js', 'nextjs', 'gatsby', 'jquery', 'bootstrap', 'tailwind',
            'asp.net', 'laravel', 'ruby on rails', 'svelte',
            
            # Databases & Data Storage
            'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'elasticsearch', 'oracle',
            'dynamodb', 'cassandra', 'neo4j', 'sqlite', 'mariadb', 'microsoft sql server',
            'sql server', 'couchdb', 'firebase', 'snowflake', 'bigquery', 'redshift',
            
            # Cloud & DevOps
            'aws', 'amazon web services', 'azure', 'microsoft azure', 'gcp', 'google cloud',
            'docker', 'kubernetes', 'k8s', 'jenkins', 'gitlab', 'github actions',
            'terraform', 'ansible', 'ci/cd', 'circleci', 'travis ci', 'cloudformation',
            'vagrant', 'puppet', 'chef', 'bamboo',
            
            # Data Science & ML & Analytics
            'machine learning', 'deep learning', 'nlp', 'natural language processing',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas', 'numpy',
            'spark', 'apache spark', 'hadoop', 'pyspark', 'jupyter', 'tableau', 'power bi',
            'looker', 'data analysis', 'data analytics', 'data visualization', 'data mining',
            'statistical analysis', 'predictive modeling', 'forecasting', 'time series',
            'regression', 'classification', 'clustering', 'neural networks', 'computer vision',
            'image processing', 'opencv', 'data warehousing', 'etl', 'big data',
            'business intelligence', 'analytics', 'quantitative analysis',
            
            # Operations & Business
            'operations management', 'process optimization', 'supply chain', 'inventory management',
            'logistics', 'lean', 'six sigma', 'kaizen', 'project management', 'agile', 'scrum',
            'kanban', 'waterfall', 'business analysis', 'business process', 'kpi', 'metrics',
            'performance management', 'quality assurance', 'quality control', 'continuous improvement',
            
            # Version Control & Collaboration
            'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial', 'version control',
            
            # Testing
            'unit testing', 'integration testing', 'selenium', 'pytest', 'junit', 'jest',
            'testing', 'test automation', 'qa', 'tdd', 'bdd',
            
            # Other technical tools
            'linux', 'unix', 'windows server', 'jira', 'confluence', 'slack', 'teams',
            'rest api', 'restful', 'graphql', 'soap', 'microservices', 'api',
            'json', 'xml', 'yaml', 'grpc', 'websocket', 'oauth', 'jwt',
            'excel', 'microsoft excel', 'google sheets', 'vba', 'macros',
            'powerpoint', 'word', 'office 365', 'google workspace',
            
            # Soft Skills & Methods (important for operations/analytics roles)
            'leadership', 'cross-functional', 'stakeholder management', 'communication',
            'problem solving', 'critical thinking', 'decision making', 'strategic planning',
            'change management', 'vendor management', 'budget management',
            'root cause analysis', 'swot analysis', 'gap analysis',
            
            # Marketing & Digital Marketing Skills
            'digital marketing', 'marketing', 'social media', 'social media marketing', 'seo', 
            'search engine optimization', 'sem', 'google ads', 'facebook ads', 'meta ads',
            'instagram marketing', 'linkedin marketing', 'twitter marketing', 'tiktok marketing',
            'email marketing', 'content marketing', 'content creation', 'copywriting',
            'marketing campaigns', 'campaign management', 'marketing automation',
            'hubspot', 'mailchimp', 'marketo', 'salesforce', 'crm',
            'google analytics', 'web analytics', 'analytics', 'data analytics',
            'market research', 'competitor analysis', 'audience analysis',
            'brand management', 'branding', 'brand strategy',
            'paid advertising', 'paid ads', 'ppc', 'pay per click',
            'conversion optimization', 'cro', 'landing pages', 'a/b testing',
            'kpi tracking', 'roi analysis', 'marketing metrics', 'reporting',
            'graphic design', 'canva', 'adobe creative suite', 'photoshop', 'illustrator',
            'figma', 'video editing', 'content strategy', 'editorial calendar',
            'influencer marketing', 'affiliate marketing', 'growth hacking',
            'meta business suite', 'facebook business manager', 'google tag manager',
            'wordpress', 'cms', 'landing page optimization',
            
            # Specific methodologies
            'agile methodology', 'scrum methodology', 'devops', 'devsecops',
            'continuous integration', 'continuous deployment', 'automation',
        }
        
        # Initialize empty set to store detected skills
        # Using set to automatically avoid duplicates
        detected_skills = set()
        
        # First pass: exact phrase matching (for multi-word skills)
        # Example: "machine learning", "natural language processing", "spring boot"
        for skill in skills_database:
            if ' ' in skill:  # Check if skill contains space (multi-word)
                # Check if entire phrase exists in text
                # Example: "machine learning" in "I have machine learning experience"
                if skill in text_lower:
                    detected_skills.add(skill)
        
        # Second pass: word boundary matching (for single-word skills)
        # Import regex module for word boundary matching
        import re
        for skill in skills_database:
            if ' ' not in skill:  # Single-word skills only (e.g., "python", "aws", "docker")
                # Use word boundaries (\b) to avoid partial matches
                # Example: "\bpython\b" matches "python" but not "pythonic" or "cpython"
                # re.escape() escapes special regex characters (e.g., "c++" → "c\+\+")
                pattern = r'\b' + re.escape(skill) + r'\b'
                
                # Search for pattern in text
                # re.search() returns match object if found, None otherwise
                if re.search(pattern, text_lower):
                    detected_skills.add(skill)
        
        # Return set of all detected skills
        # Set automatically contains no duplicates
        return detected_skills
    
    def extract_keywords(self, text: str, top_n: int = 30) -> Dict[str, List[str]]:
        """
        Extract keywords using multiple methods
        
        Args:
            text: Text to extract keywords from
            top_n: Number of top keywords to return per method
            
        Returns:
            Dictionary with keywords from different methods
        """
        # Initialize result dictionary to store keywords from each method
        result = {
            # Method 1: Extract technical skills using hardcoded skills database (~200+ skills)
            # Returns: ['python', 'aws', 'docker', 'machine learning', ...]
            'technical_skills': list(self.extract_technical_skills(text)),
            
            # Method 2: Extract keywords using TF-IDF (Term Frequency-Inverse Document Frequency)
            # Returns top N keywords based on statistical importance
            # Returns: ['developer', 'software engineering', 'experience', ...]
            'tfidf_keywords': self.extract_keywords_tfidf(text, top_n),
        }
        
        # Method 3: Extract keywords using spaCy NLP (if available)
        # This uses Named Entity Recognition and POS tagging
        if self.use_spacy and self.nlp:
            # Returns: ['python', 'google', 'senior developer', ...]
            result['spacy_keywords'] = self.extract_keywords_spacy(text, top_n)
        
        # Combine all keywords from all methods into a single set
        # This creates a comprehensive list without duplicates
        all_keywords = set()
        for keywords in result.values():
            # update() adds all items from the list to the set
            all_keywords.update(keywords)
        
        # Add combined keywords to result dictionary
        # Convert set back to list for JSON serialization
        result['all_keywords'] = list(all_keywords)
        
        # Return dictionary with keywords from all methods
        # Structure: {
        #     'technical_skills': [...],
        #     'tfidf_keywords': [...],
        #     'spacy_keywords': [...],  (optional)
        #     'all_keywords': [...]
        # }
        return result
    
    def extract_cv_entities(self, text: str) -> Dict:
        """
        Extract structured CV information using enhanced NLP
        Similar to en_cv_info_extr functionality
        
        Args:
            text: Resume text
            
        Returns:
            Dictionary with extracted CV entities
        """
        # If spaCy model is not loaded, return empty dictionary
        if not self.nlp:
            return {}
        
        # Process text through spaCy's NLP pipeline
        # Performs: tokenization, POS tagging, NER, lemmatization, dependency parsing
        doc = self.nlp(text)
        
        # Initialize dictionary to store different types of extracted entities
        entities = {
            'skills': [],           # Technical skills (Python, AWS, etc.)
            'education': [],        # Education-related information (degrees, universities)
            'experience_years': [], # Years of experience extracted from text
            'organizations': [],    # Company/organization names
            'locations': [],        # Geographic locations (cities, countries)
            'certifications': [],   # Certifications and licenses
            'tools': [],            # Tools and software mentioned
            'languages': [],        # Programming/spoken languages
            'dates': []             # Dates mentioned (employment dates, graduation dates)
        }
        
        # Step 1: Extract named entities using spaCy's NER
        # doc.ents contains all entities detected by the NER model
        for ent in doc.ents:
            # ORG label: Organizations/companies (e.g., "Google", "Microsoft", "Stanford University")
            if ent.label_ == 'ORG':
                entities['organizations'].append(ent.text)
            # GPE label: Geopolitical entities (e.g., "New York", "California", "USA")
            # LOC label: Non-GPE locations (e.g., "Mount Everest", "Pacific Ocean")
            elif ent.label_ == 'GPE' or ent.label_ == 'LOC':
                entities['locations'].append(ent.text)
            # DATE label: Dates and date ranges (e.g., "2020-2023", "January 2022", "last 5 years")
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
        
        # Step 2: Extract technical skills using comprehensive skills database
        # This uses the hardcoded database of 200+ technical skills
        entities['skills'] = list(self.extract_technical_skills(text))
        
        # Step 3: Extract education keywords and context
        # Define common education-related terms
        education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'associate', 'diploma',
            'degree', 'university', 'college', 'institute', 'school',
            'bs', 'ba', 'ms', 'ma', 'mba', 'bsc', 'msc', 'beng', 'meng'
        ]
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Search for education keywords in text
        for keyword in education_keywords:
            if keyword in text_lower:
                # Find context around the keyword (sentences/lines containing it)
                # Import regex module
                import re
                # Pattern: capture everything before and after keyword until newline or period
                # [^\n.]* = match any character except newline or period
                pattern = rf'([^\n.]*{keyword}[^\n.]*)'
                # Find all matches (case-insensitive)
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                # Limit to top 3 matches to avoid clutter
                entities['education'].extend(matches[:3])
        
        # Step 4: Extract certification keywords and context
        # Define common certification-related terms
        cert_keywords = [
            'certified', 'certification', 'certificate', 'licensed', 'license',
            'aws', 'azure', 'google cloud', 'pmp', 'cissp', 'comptia', 'itil',
            'scrum master', 'csm', 'cpa', 'cfa', 'phr', 'sphr'
        ]
        
        # Search for certification keywords in text
        for keyword in cert_keywords:
            if keyword in text_lower:
                # Find context around the keyword (sentences/lines containing it)
                import re
                pattern = rf'([^\n.]*{keyword}[^\n.]*)'
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                # Limit to top 3 matches per keyword
                entities['certifications'].extend(matches[:3])
        
        # Step 5: Extract years of experience
        # Look for patterns like "5 years", "5+ years", "5-7 years", "5 yrs"
        # Regex pattern breakdown:
        # (\d+) = capture one or more digits (first number)
        # [\+\-\s]* = optional plus, minus, or space characters
        # (?:to|-)? = optional "to" or "-" (non-capturing group)
        # \s* = optional whitespace
        # (\d+)? = optional second number (for ranges like "5-7")
        # \s* = optional whitespace
        # (?:years?|yrs?) = "year", "years", "yr", or "yrs" (non-capturing group)
        exp_pattern = r'(\d+)[\+\-\s]*(?:to|-)?\s*(\d+)?\s*(?:years?|yrs?)'
        exp_matches = re.findall(exp_pattern, text_lower)
        
        # Extract the first number from each match (minimum years)
        for match in exp_matches:
            if match[0]:  # If first number exists
                entities['experience_years'].append(match[0])
        
        # Step 6: Remove duplicates from all entity lists
        # Iterate through each category in entities dictionary
        for key in entities:
            if isinstance(entities[key], list):
                # Convert to set (removes duplicates) then back to list
                # Example: ['python', 'aws', 'python'] → {'python', 'aws'} → ['python', 'aws']
                entities[key] = list(set(entities[key]))
        
        # Return dictionary with all extracted entities
        # Structure: {
        #     'skills': ['python', 'aws', ...],
        #     'education': ['bachelor degree in computer science', ...],
        #     'experience_years': ['5', '3', ...],
        #     'organizations': ['Google', 'Microsoft', ...],
        #     'locations': ['San Francisco', 'California', ...],
        #     'certifications': ['aws certified solutions architect', ...],
        #     'tools': [],
        #     'languages': [],
        #     'dates': ['2020-2023', 'January 2022', ...]
        # }
        return entities
