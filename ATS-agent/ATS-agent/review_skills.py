"""
LLM-Based Skills Validator
Reviews skills and marks non-skills for removal using AI judgment
"""

import pandas as pd
import time

def review_top_skills():
    """Review top 200 skills manually"""
    df = pd.read_csv('data/skills_cleaned_v2.csv')
    
    print("Top 200 Skills Review:")
    print("=" * 80)
    
    for i in range(min(200, len(df))):
        skill = df.iloc[i]['skill']
        freq = df.iloc[i]['frequency']
        print(f"{i+1:3d}. [{freq:4d}x] {skill}")

if __name__ == "__main__":
    review_top_skills()
