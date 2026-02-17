"""
RAG-based Skills Extraction Module
Uses embeddings and vector similarity to extract skills from resumes and job descriptions
Loads skills from the provided CSV dataset
"""

import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple
from pathlib import Path
import re
from tqdm import tqdm


class RAGSkillsExtractor:
    """Extract skills using RAG (Retrieval-Augmented Generation) with CSV dataset"""
    
    def __init__(
        self, 
        skills_csv_path: str = r'C:\Users\Admin\Documents\ATS-agent\data\skills_exploded (2).csv',
        embedding_model: str = 'all-MiniLM-L6-v2',
        max_skills: int = None
    ):
        """
        Initialize RAG skills extractor
        
        Args:
            skills_csv_path: Path to CSV file containing skills
            embedding_model: Name of the sentence transformer model to use
            max_skills: Maximum number of skills to load (None = all). Use smaller number for faster loading.
        """
        # Store the path to the CSV file containing skills database
        self.skills_csv_path = skills_csv_path
        
        # Store the name of the Sentence Transformer model to use (all-MiniLM-L6-v2)
        self.embedding_model_name = embedding_model
        
        # Store maximum number of skills to load (for performance tuning)
        self.max_skills = max_skills
        
        # Initialize model placeholder (will be loaded in _initialize_model)
        self.model = None
        
        # Initialize skills list placeholder (will be loaded from CSV)
        self.skills_list = None
        
        # Initialize embeddings array placeholder (will be computed or loaded from cache)
        self.skill_embeddings = None
        
        # Create cache file name based on max_skills setting
        # If max_skills=10000: 'skills_embeddings_csv_10000.pkl'
        # If max_skills=None: 'skills_embeddings_csv_full.pkl'
        cache_suffix = f'_{max_skills}' if max_skills else '_full'
        self.embeddings_cache_path = Path(f'skills_embeddings_csv{cache_suffix}.pkl')
        
        # Step 1: Load the Sentence Transformer model
        self._initialize_model()
        
        # Step 2: Load skills from CSV file
        self._load_skills_from_csv()
        
        # Step 3: Load pre-computed embeddings from cache or create new ones
        self._load_or_create_embeddings()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            # Import the SentenceTransformer class from sentence_transformers library
            from sentence_transformers import SentenceTransformer
            
            # Print status message to inform user which model is being loaded
            print(f"Loading embedding model: {self.embedding_model_name}...")
            
            # Load the pre-trained Sentence Transformer model
            # This downloads the model if not cached (~90MB for all-MiniLM-L6-v2)
            # Model has 22M parameters, 6 transformer layers, outputs 384-dimensional vectors
            self.model = SentenceTransformer(self.embedding_model_name)
            
            # Confirm successful loading
            print("Model loaded successfully")
            
        except ImportError:
            # Handle case where sentence-transformers package is not installed
            print("ERROR: sentence-transformers not installed.")
            print("Install with: pip install sentence-transformers")
            raise  # Re-raise the exception to stop execution
            
        except Exception as e:
            # Handle any other errors during model loading
            print(f"Error loading model: {e}")
            raise  # Re-raise the exception to stop execution
    
    def _load_skills_from_csv(self):
        """Load skills from CSV file"""
        # Print the CSV file path being loaded
        print(f"Loading skills from CSV: {self.skills_csv_path}")
        
        try:
            # Read CSV file with pandas
            if self.max_skills:
                # If max_skills is set, only read the first N rows for performance
                # Example: max_skills=10000 reads only first 10,000 rows from 3.2M total
                df = pd.read_csv(self.skills_csv_path, nrows=self.max_skills)
                print(f"Loaded {len(df)} skills (limited to {self.max_skills})")
            else:
                # If max_skills=None, load all rows from the CSV file
                df = pd.read_csv(self.skills_csv_path)
                print(f"Loaded {len(df)} skills from CSV")
            
            # Extract skills from first column, remove null values, get unique skills
            # iloc[:, 0] = select first column (job_skills column)
            # dropna() = remove empty/null values
            # unique() = remove duplicates
            # tolist() = convert to Python list
            skills = df.iloc[:, 0].dropna().unique().tolist()
            
            # Clean the skills list by filtering out invalid entries
            cleaned_skills = []
            for skill in skills:
                # Convert to string and remove leading/trailing whitespace
                skill_str = str(skill).strip()
                
                # Keep skills that are:
                # 1. At least 2 characters long (keeps "C+", "R", "Go" but removes "a", "1")
                # 2. Not purely numeric (removes "123", "2023" but keeps "C++", "Python3")
                if len(skill_str) >= 2 and not skill_str.isdigit():
                    cleaned_skills.append(skill_str)
            
            # Store the cleaned skills list as instance variable
            self.skills_list = cleaned_skills
            
            # Print the final count of unique skills after cleaning
            print(f"{len(self.skills_list)} unique skills after cleaning")
            
        except Exception as e:
            # Handle any errors during CSV loading (file not found, corrupt data, etc.)
            print(f"Error loading CSV: {e}")
            raise  # Re-raise to stop execution
    
    def _load_or_create_embeddings(self):
        """Load existing embeddings or create new ones"""
        # Check if cache file exists (e.g., 'skills_embeddings_csv_10000.pkl')
        if self.embeddings_cache_path.exists():
            print(f"Loading cached skill embeddings from {self.embeddings_cache_path}...")
            try:
                # Open the pickle file in binary read mode
                with open(self.embeddings_cache_path, 'rb') as f:
                    # Deserialize the cached data (contains skills list and embeddings array)
                    cache_data = pickle.load(f)
                    
                    # Extract the embeddings numpy array (shape: N x 384)
                    self.skill_embeddings = cache_data['embeddings']
                    
                    # Extract the skills list to verify it matches current skills
                    cached_skills = cache_data['skills']
                    
                    # Verify that cached skills exactly match the current skills list
                    # This prevents using outdated embeddings with different skills
                    if cached_skills == self.skills_list:
                        print(f"Loaded embeddings for {len(self.skills_list)} skills from cache")
                        return  # Successfully loaded from cache, exit function
                    else:
                        # Skills don't match, need to regenerate embeddings
                        print("Cache does not match current skills, regenerating...")
                        
            except Exception as e:
                # Handle errors reading cache file (corrupt file, wrong format, etc.)
                print(f"Warning: Could not load cache: {e}")
        
        # If we reach here, either cache doesn't exist or needs regeneration
        print("Creating skill embeddings (this may take a while)...")
        
        # Create embeddings by encoding all skills with the Sentence Transformer model
        self._create_embeddings()
        
        # Save the newly created embeddings to cache file for future use
        self._save_embeddings()
    
    def _create_embeddings(self):
        """Create embeddings for all skills"""
        # Print the total number of skills that need to be encoded
        print(f"Encoding {len(self.skills_list)} skills...")
        
        # Process skills in batches for memory efficiency
        # Batch size of 1000 balances memory usage and processing speed
        batch_size = 1000
        
        # List to store embeddings from each batch
        embeddings_list = []
        
        # Iterate through skills in batches with progress bar
        # range(0, N, 1000) generates: 0, 1000, 2000, 3000, ... up to N
        for i in tqdm(range(0, len(self.skills_list), batch_size), desc="Encoding skills"):
            # Extract current batch of skills (e.g., skills 0-999, 1000-1999, etc.)
            batch = self.skills_list[i:i+batch_size]
            
            # Encode the batch using Sentence Transformer model
            # Each skill becomes a 384-dimensional vector
            # show_progress_bar=False because tqdm already shows progress
            # Example: "Python" → [0.023, -0.156, 0.089, ..., 0.234] (384 numbers)
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            
            # Append the batch embeddings to the list
            embeddings_list.append(batch_embeddings)
        
        # Stack all batch embeddings vertically to create final array
        # If we have 10 batches of (1000, 384), vstack creates (10000, 384)
        self.skill_embeddings = np.vstack(embeddings_list)
        
        # Print final shape for confirmation (e.g., "(9988, 384)")
        print(f"Created embeddings with shape: {self.skill_embeddings.shape}")
    
    def _save_embeddings(self):
        """Save embeddings to cache file"""
        try:
            # Create a dictionary containing both skills list and embeddings array
            # This allows us to verify the cache matches the skills later
            cache_data = {
                'skills': self.skills_list,        # List of 9,988 skill names
                'embeddings': self.skill_embeddings  # NumPy array of shape (9988, 384)
            }
            
            # Open cache file in binary write mode
            with open(self.embeddings_cache_path, 'wb') as f:
                # Serialize and save the data structure to pickle file
                # This creates a ~15MB file for 10,000 skills
                pickle.dump(cache_data, f)
                
            # Confirm successful save
            print(f"Saved embeddings to {self.embeddings_cache_path}")
            
        except Exception as e:
            # Handle errors during save (disk full, permission denied, etc.)
            # This is a warning, not critical error - embeddings already in memory
            print(f"Warning: Could not save embeddings cache: {e}")
    
    def _extract_ngrams(self, text: str, n_range: Tuple[int, int] = (1, 5)) -> List[str]:
        """
        Extract n-grams from text
        
        Args:
            text: Input text
            n_range: Tuple of (min_n, max_n) for n-gram extraction
            
        Returns:
            List of n-grams
        """
        # Clean text using regex to preserve important technical characters
        # Keeps: letters, numbers, spaces, periods, hyphens, plus signs, hashtags, parentheses, slashes, ampersands
        # This preserves: "C++", "C#", ".NET", "Node.js", "React/Redux", "AWS & Azure"
        # Removes: quotes, special characters, punctuation that's not needed
        text = re.sub(r'[^\w\s\.\-\+\#\(\)\/\&]', ' ', text)
        
        # Split text into individual words by whitespace
        # Example: "Python developer with AWS" → ["Python", "developer", "with", "AWS"]
        words = text.split()
        
        # Initialize list to store all n-grams
        ngrams = []
        
        # Generate n-grams for each size from min_n to max_n (inclusive)
        # n_range=(1, 5) generates: 1-grams, 2-grams, 3-grams, 4-grams, 5-grams
        for n in range(n_range[0], n_range[1] + 1):
            # For each possible starting position in the words list
            # len(words) - n + 1 ensures we don't go out of bounds
            # Example with 4 words and n=2: positions 0, 1, 2 (can create 3 bigrams)
            for i in range(len(words) - n + 1):
                # Extract n consecutive words starting at position i
                # Join them with spaces to create the n-gram
                # Example: words[0:2] = ["Python", "developer"] → "Python developer"
                ngram = ' '.join(words[i:i+n])
                
                # Filter out very short n-grams (single character)
                # This removes meaningless n-grams like "a", "I", etc.
                if len(ngram) > 1:
                    ngrams.append(ngram)
        
        # Return the complete list of n-grams
        # Example output: ["python", "developer", "with", "aws", "python developer", 
        #                  "developer with", "with aws", "python developer with", ...]
        return ngrams
    
    def extract_skills_rag(
        self, 
        text: str, 
        threshold: float = 0.6,
        top_k: int = None,
        return_scores: bool = False
    ) -> List[str] | List[Tuple[str, float]]:
        """
        Extract skills using RAG approach with semantic similarity
        
        Args:
            text: Input text (resume or job description)
            threshold: Minimum similarity threshold (0-1). Higher = stricter matching
            top_k: If set, return only top k matches regardless of threshold
            return_scores: If True, return (skill, score) tuples
            
        Returns:
            List of detected skills or list of (skill, score) tuples
        """
        # Step 1: Extract n-grams (1-5 word chunks) from the input text
        # Example: "Python developer" → ["python", "developer", "python developer"]
        ngrams = self._extract_ngrams(text)
        
        # Handle empty case - if no n-grams extracted, return empty list
        if not ngrams:
            return [] if not return_scores else []
        
        # Step 2: Encode all n-grams into 384-dimensional vectors
        print(f"Encoding {len(ngrams)} text segments...")
        # Each n-gram becomes a vector for semantic comparison
        # Example: "python" → [0.234, -0.156, 0.089, ..., 0.123] (384 numbers)
        ngram_embeddings = self.model.encode(ngrams, show_progress_bar=False)
        
        # Step 3: Import cosine similarity function for vector comparison
        from sklearn.metrics.pairwise import cosine_similarity
        
        print("Computing similarity scores...")
        # Step 4: Calculate similarity between ALL n-grams and ALL skills
        # Creates a matrix: (num_ngrams × num_skills)
        # Example: (50 ngrams × 9988 skills) = 499,400 similarity scores
        # Each cell contains cosine similarity score (0.0 to 1.0)
        similarities = cosine_similarity(ngram_embeddings, self.skill_embeddings)
        
        # Step 5: For each skill, find the HIGHEST similarity score across all n-grams
        # axis=0 means take maximum along the n-grams dimension (columns)
        # Result: array of 9,988 scores (one per skill)
        # Example: "Python" skill might match "python" (0.95), "python developer" (0.88)
        #          → we take 0.95 as the final score
        max_similarities = np.max(similarities, axis=0)
        
        # Step 6: Filter skills by threshold and collect detected skills
        detected_skills = []
        # Loop through each skill with its maximum similarity score
        for idx, score in enumerate(max_similarities):
            # Only keep skills that meet or exceed the threshold (default 0.6 = 60% similarity)
            if score >= threshold:
                # Get the skill name from the skills list
                skill = self.skills_list[idx]
                # Store as tuple: (skill_name, similarity_score)
                detected_skills.append((skill, float(score)))
        
        # Step 7: Sort skills by similarity score (highest first)
        # lambda x: x[1] means sort by the second element of tuple (the score)
        # reverse=True means descending order (best matches first)
        detected_skills.sort(key=lambda x: x[1], reverse=True)
        
        # Step 8: Apply top_k limit if specified
        # If top_k=20, only return the top 20 skills regardless of how many passed threshold
        if top_k:
            detected_skills = detected_skills[:top_k]
        
        # Print summary of detected skills
        print(f"Found {len(detected_skills)} skills above threshold {threshold}")
        
        # Step 9: Return results in requested format
        if return_scores:
            # Return list of tuples: [("Python", 0.95), ("AWS", 0.87), ...]
            return detected_skills
        else:
            # Return only skill names: ["Python", "AWS", "TensorFlow", ...]
            return [skill for skill, _ in detected_skills]
    
    def compare_skills(
        self, 
        resume_text: str, 
        job_desc_text: str,
        threshold: float = 0.6
    ) -> Dict:
        """
        Compare skills from resume and job description
        
        Args:
            resume_text: Resume text
            job_desc_text: Job description text
            threshold: Minimum similarity threshold
            
        Returns:
            Dictionary with matched, missing, and additional skills
        """
        print("\n" + "="*80)
        print("EXTRACTING SKILLS FROM RESUME")
        print("="*80)
        resume_skills = set(self.extract_skills_rag(resume_text, threshold=threshold))
        
        print("\n" + "="*80)
        print("EXTRACTING SKILLS FROM JOB DESCRIPTION")
        print("="*80)
        job_skills = set(self.extract_skills_rag(job_desc_text, threshold=threshold))
        
        matched = resume_skills & job_skills
        missing = job_skills - resume_skills
        additional = resume_skills - job_skills
        
        return {
            'matched_skills': sorted(list(matched)),
            'missing_skills': sorted(list(missing)),
            'additional_skills': sorted(list(additional)),
            'match_percentage': len(matched) / len(job_skills) * 100 if job_skills else 0,
            'resume_skill_count': len(resume_skills),
            'job_skill_count': len(job_skills)
        }
    
    def get_skill_recommendations(
        self, 
        current_skills: List[str], 
        target_role: str,
        top_n: int = 10,
        threshold: float = 0.6
    ) -> List[Tuple[str, float]]:
        """
        Get recommended skills based on current skills and target role
        
        Args:
            current_skills: List of current skills
            target_role: Target job role or description
            top_n: Number of recommendations to return
            threshold: Minimum relevance threshold
            
        Returns:
            List of (skill, relevance_score) tuples
        """
        # Extract skills from target role
        target_skills = set(self.extract_skills_rag(target_role, threshold=threshold))
        
        # Skills that are in target but not in current
        recommended = target_skills - set(current_skills)
        
        # Encode target role
        role_embedding = self.model.encode([target_role])[0]
        
        # Calculate similarity between role and recommended skills
        from sklearn.metrics.pairwise import cosine_similarity
        
        recommendations = []
        for skill in recommended:
            skill_idx = self.skills_list.index(skill) if skill in self.skills_list else None
            if skill_idx is not None:
                skill_embedding = self.skill_embeddings[skill_idx]
                relevance = cosine_similarity([role_embedding], [skill_embedding])[0][0]
                recommendations.append((skill, float(relevance)))
        
        # Sort by relevance and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG-based Skills Extractor using CSV dataset')
    parser.add_argument('--max-skills', type=int, default=50000, 
                       help='Maximum number of skills to load (default: 50000, use None for all)')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Similarity threshold 0-1 (default: 0.6)')
    parser.add_argument('--test', action='store_true',
                       help='Run test with sample text')
    
    args = parser.parse_args()
    
    print("="*80)
    print("RAG SKILLS EXTRACTOR - CSV DATASET")
    print("="*80)
    
    # Initialize extractor
    extractor = RAGSkillsExtractor(max_skills=args.max_skills)
    
    if args.test:
        # Example resume text
        resume_text = """
        Senior Software Engineer with 5 years of experience in Python and JavaScript.
        Proficient in React, Django, and PostgreSQL. Experience with AWS and Docker.
        Strong background in machine learning and data analysis using pandas and scikit-learn.
        Led cross-functional teams in agile environment. Bachelor's degree in Computer Science.
        """
        
        # Example job description
        job_desc = """
        Looking for a Full Stack Developer with expertise in React, Node.js, and MongoDB.
        Experience with cloud platforms (AWS or Azure) required.
        Knowledge of Docker and Kubernetes is a plus.
        Strong problem-solving skills and ability to work in agile teams.
        Bachelor's degree in Computer Science or related field required.
        """
        
        print("\n" + "="*80)
        print("TEST: EXTRACTING SKILLS FROM RESUME")
        print("="*80)
        
        resume_skills = extractor.extract_skills_rag(
            resume_text, 
            threshold=args.threshold, 
            return_scores=True
        )
        print(f"\nFound {len(resume_skills)} skills:\n")
        for skill, score in resume_skills[:20]:
            print(f"  • {skill:50} (score: {score:.3f})")
        if len(resume_skills) > 20:
            print(f"\n  ... and {len(resume_skills) - 20} more skills")
        
        print("\n" + "="*80)
        print("TEST: COMPARING RESUME WITH JOB DESCRIPTION")
        print("="*80)
        
        comparison = extractor.compare_skills(resume_text, job_desc, threshold=args.threshold)
        
        print(f"\n✓ Matched Skills ({len(comparison['matched_skills'])}):")
        for skill in comparison['matched_skills']:
            print(f"  • {skill}")
        
        print(f"\n✗ Missing Skills ({len(comparison['missing_skills'])}):")
        for skill in comparison['missing_skills']:
            print(f"  • {skill}")
        
        print(f"\n📊 Match Percentage: {comparison['match_percentage']:.1f}%")
        
        print("\n" + "="*80)
        print("TEST: SKILL RECOMMENDATIONS")
        print("="*80)
        
        current_skills = [skill for skill, _ in resume_skills]
        recommendations = extractor.get_skill_recommendations(
            current_skills, 
            job_desc, 
            top_n=10,
            threshold=args.threshold
        )
        
        print("\nTop 10 recommended skills to learn:\n")
        for skill, score in recommendations:
            print(f"  • {skill:50} (relevance: {score:.3f})")
    
    print("\n" + "="*80)
    print("Extractor ready for use!")
    print("="*80)


if __name__ == "__main__":
    main()
