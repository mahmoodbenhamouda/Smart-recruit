"""
Analyze the training data to understand why the model predicts "Project Manager" for everything
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

print("Loading dataset...")
df = pd.read_csv('../../deep_Learning_Project/resume_screening_dataset_train.csv')

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Check class distribution
print("\n" + "="*80)
print("CLASS DISTRIBUTION")
print("="*80)
role_counts = df['Role'].value_counts()
print(f"\nTotal unique roles: {len(role_counts)}")
print(f"\nTop 10 most frequent roles:")
for role, count in role_counts.head(10).items():
    print(f"  {role:40s}: {count:4d} ({count/len(df)*100:.1f}%)")

# Check if resumes contain role-specific keywords
print("\n" + "="*80)
print("CHECKING RESUME CONTENT")
print("="*80)

# Sample a few resumes from different roles
sample_roles = ['Data Scientist', 'Software Engineer', 'Product Manager', 'DevOps Engineer']
for role in sample_roles:
    if role in df['Role'].values:
        sample = df[df['Role'] == role].iloc[0]
        resume_text = str(sample['Resume']).lower()
        print(f"\n{role}:")
        print(f"  Resume length: {len(resume_text)} chars")
        # Check if role name appears in resume
        if role.lower() in resume_text:
            print(f"  ✅ Role name appears in resume")
        else:
            print(f"  ❌ Role name does NOT appear in resume")
        # Show a snippet
        print(f"  Snippet: {resume_text[200:400]}")

# Check for discriminative features
print("\n" + "="*80)
print("FEATURE ANALYSIS")
print("="*80)

# Use TF-IDF to find most important features for each role
print("\nExtracting TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
X = vectorizer.fit_transform(df['Resume'].astype(str))

feature_names = vectorizer.get_feature_names_out()

# For top roles, find their most distinctive features
top_roles = role_counts.head(5).index.tolist()

for role in top_roles:
    role_mask = df['Role'] == role
    role_vectors = X[role_mask].toarray()
    non_role_vectors = X[~role_mask].toarray()
    
    role_mean = role_vectors.mean(axis=0)
    non_role_mean = non_role_vectors.mean(axis=0)
    
    # Features that appear more in this role than others
    diff = role_mean - non_role_mean
    top_features_idx = np.argsort(diff)[-10:][::-1]
    
    print(f"\n{role}:")
    print("  Top distinctive features:")
    for idx in top_features_idx:
        if diff[idx] > 0:
            print(f"    - {feature_names[idx]}: {diff[idx]:.4f}")

# Check if there's enough variation in resumes
print("\n" + "="*80)
print("RESUME VARIATION ANALYSIS")
print("="*80)

# Check how similar resumes are within and across roles
from sklearn.metrics.pairwise import cosine_similarity

print("\nCalculating resume similarities...")
# Sample 100 resumes for speed
sample_size = min(100, len(df))
sample_df = df.sample(n=sample_size, random_state=42)
sample_X = vectorizer.transform(sample_df['Resume'].astype(str))

similarities = cosine_similarity(sample_X)

# Within-role similarity
within_role_sims = []
for role in sample_df['Role'].unique():
    role_idx = sample_df[sample_df['Role'] == role].index.tolist()
    if len(role_idx) > 1:
        for i in range(len(role_idx)):
            for j in range(i+1, len(role_idx)):
                idx_i = sample_df.index.get_loc(role_idx[i])
                idx_j = sample_df.index.get_loc(role_idx[j])
                within_role_sims.append(similarities[idx_i, idx_j])

# Across-role similarity
across_role_sims = []
for i in range(len(sample_df)):
    for j in range(i+1, len(sample_df)):
        if sample_df.iloc[i]['Role'] != sample_df.iloc[j]['Role']:
            across_role_sims.append(similarities[i, j])

if within_role_sims:
    print(f"\nAverage within-role similarity: {np.mean(within_role_sims):.4f}")
if across_role_sims:
    print(f"Average across-role similarity: {np.mean(across_role_sims):.4f}")

if within_role_sims and across_role_sims:
    if np.mean(within_role_sims) <= np.mean(across_role_sims):
        print("\n❌ WARNING: Resumes are MORE similar across roles than within roles!")
        print("   This means the dataset may not have enough discriminative features.")
    else:
        print("\n✅ Resumes within the same role are more similar (good!)")
