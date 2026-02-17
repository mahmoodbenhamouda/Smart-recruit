"""
Create a skills-based training dataset from resumes
Extract only technical skills and keywords, removing boilerplate
"""
import pandas as pd
import re
import pickle
from pathlib import Path

def extract_skills_section(resume_text):
    """
    Extract skills/keywords from resume, removing common boilerplate
    """
    text = str(resume_text).lower()
    
    # Try to find skills section explicitly
    skills_match = re.search(r'(?:skills?|technologies?|technical skills?|core competencies):?\s*([^\n]*(?:\n[^\n]*){0,20})', text, re.IGNORECASE)
    
    if skills_match:
        skills_text = skills_match.group(1)
    else:
        # If no explicit skills section, use the whole resume but clean it heavily
        skills_text = text
    
    # Remove common boilerplate phrases
    boilerplate = [
        r'professional summary',
        r'work experience',
        r'education',
        r'contact information',
        r'email:?\s*\S+',
        r'phone:?\s*[\d\-\(\) ]+',
        r'linkedin:?\s*\S+',
        r'github:?\s*\S+',
        r'results?-driven',
        r'highly motivated',
        r'strong background',
        r'proven track record',
        r'team player',
        r'\d+\+?\s*years? of experience',
        r'bachelor\'s? degree',
        r'master\'s? degree',
        r'university',
        r'[a-z]+\.com',
        r'\([0-9]{3}\)\s*[0-9]{3}-[0-9]{4}',
        r'here\'s a professional resume for',
        r'professional resume',
        r'summary:?',
        r'objective:?',
    ]
    
    for pattern in boilerplate:
        skills_text = re.sub(pattern, ' ', skills_text, flags=re.IGNORECASE)
    
    # Keep only alphanumeric, plus, hash, dot, hyphen
    skills_text = re.sub(r'[^a-z0-9\s\+\#\.\-]', ' ', skills_text)
    
    # Remove standalone numbers and very short words
    skills_text = re.sub(r'\b\d+\b', ' ', skills_text)
    skills_text = re.sub(r'\b[a-z]\b', ' ', skills_text)
    
    # Clean up whitespace
    skills_text = ' '.join(skills_text.split())
    
    return skills_text

print("="*80)
print("CREATING SKILLS-BASED TRAINING DATASET")
print("="*80)

# Load original dataset
dataset_path = Path("../../deep_Learning_Project/resume_screening_dataset_train.csv")
print(f"\nLoading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)

print(f"Original dataset: {len(df)} samples, {df['Role'].nunique()} unique roles")

# Extract skills from resumes
print("\nExtracting skills from resumes...")
df['Skills'] = df['Resume'].apply(extract_skills_section)

# Show examples
print("\n" + "="*80)
print("BEFORE & AFTER EXAMPLES")
print("="*80)

for i in range(min(3, len(df))):
    role = df.iloc[i]['Role']
    original = df.iloc[i]['Resume'][:300]
    extracted = df.iloc[i]['Skills'][:300]
    
    print(f"\n{i+1}. Role: {role}")
    print(f"   Original (300 chars): {original}...")
    print(f"   Extracted (300 chars): {extracted}...")
    print("-"*80)

# Check average lengths
avg_original = df['Resume'].str.len().mean()
avg_skills = df['Skills'].str.len().mean()
print(f"\nAverage length:")
print(f"  Original resumes: {avg_original:.0f} characters")
print(f"  Extracted skills: {avg_skills:.0f} characters ({avg_skills/avg_original*100:.1f}% of original)")

# Save the skills-based dataset
output_path = Path("resume_skills_train.csv")
df[['Role', 'Skills']].to_csv(output_path, index=False)
print(f"\n✅ Saved skills-based dataset to: {output_path}")
print(f"   Columns: Role, Skills")
print(f"   Size: {len(df)} samples")

#Show class distribution
print("\n" + "="*80)
print("CLASS DISTRIBUTION")
print("="*80)
class_counts = df['Role'].value_counts()
print(f"\nTop 15 most frequent roles:")
for role, count in class_counts.head(15).items():
    print(f"  {role:40s}: {count:4d} samples")

print(f"\n✅ Done! Use 'resume_skills_train.csv' for training.")
