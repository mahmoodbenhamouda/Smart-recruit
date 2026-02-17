"""
AI-Assisted Skills Cleaner - Final Pass
Removes remaining non-skills identified through review
"""

import pandas as pd
import re

# US States (locations, not skills)
US_STATES = {
    'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
    'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
    'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
    'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
    'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
    'new hampshire', 'new jersey', 'new mexico', 'new york',
    'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon',
    'pennsylvania', 'rhode island', 'south carolina', 'south dakota',
    'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
    'west virginia', 'wisconsin', 'wyoming'
}

# Additional non-skills to remove
NON_SKILLS_TO_REMOVE = {
    # Benefits
    'paid time off', 'pto', 'vision', 'insurance', 'retirement plan',
    'eap', 'hsa', 'fsa', 'flexible schedule', 'vacation',
    
    # Employment terms
    'fulltime', 'full time', 'part time', 'parttime', 'contract',
    'temporary', 'permanent', 'availability', 'doe', 'minimum',
    
    # Generic adjectives/adverbs
    'helpful', 'advanced', 'intermediate', 'basic', 'minimum',
    'maximum', 'excellent', 'good', 'strong', 'preferred',
    'required', 'optional', 'desirable', 'desired', 'advantageous',
    
    # Generic nouns
    'work experience', 'experience', 'development', 'technology',
    'reporting', 'projects', 'policies', 'plans', 'qualification',
    'accreditation', 'licensing', 'vaccination', 'medical',
    
    # Physical states/actions
    'standing', 'lifting', 'walking', 'sitting', 'bending',
    'lifting ability', 'physical strength',
    
    # Requirement phrases
    'not required', 'if required', 'an asset', 'a plus',
    'verbal and written', 'written and verbal', 'oral and written',
    'written and oral',
    
    # Numbers/vague terms
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    
    # Duplicates with formatting issues
    "bachelor''s degree",  # Has double apostrophe - already have bachelor's degree
}

def is_location(skill: str) -> bool:
    """Check if skill is a location (state, country, etc.)"""
    skill_lower = skill.lower().strip()
    
    # US States
    if skill_lower in US_STATES:
        return True
    
    # State abbreviations (but exclude valid skills)
    valid_abbrevs = {
        'sql', 'api', 'aws', 'gcp', 'sap', 'erp', 'crm', 'ehr', 'emr', 
        'cad', 'qa', 'ml', 'ai', 'bi', 'ui', 'ux', 'hr', 'it'
    }
    if len(skill_lower) == 2 and skill_lower not in valid_abbrevs:
        # Could be state abbreviation
        return True
    
    return False

def is_benefit_or_employment_term(skill: str) -> bool:
    """Check if skill is actually a benefit or employment term"""
    skill_lower = skill.lower().strip()
    
    benefit_keywords = [
        'time off', 'insurance', 'benefit', 'retirement', 'pension',
        'vacation', 'holiday', 'pto', 'sick', 'leave', 'bonus',
        'schedule', 'shift', 'hours', 'fulltime', 'full time',
        'part time', 'parttime', 'contract', 'temporary', 'permanent'
    ]
    
    for keyword in benefit_keywords:
        if keyword in skill_lower:
            return True
    
    return False

def is_generic_term(skill: str) -> bool:
    """Check if skill is too generic"""
    skill_lower = skill.lower().strip()
    
    # Just a number
    if skill_lower.isdigit() and int(skill_lower) < 20:
        return True
    
    # Single generic words
    generic_words = {
        'helpful', 'minimum', 'maximum', 'doe', 'availability',
        'experience', 'development', 'technology', 'reporting',
        'projects', 'policies', 'plans', 'medical', 'qualification',
        'accreditation', 'licensing', 'vaccination', 'standing',
        'advanced', 'intermediate', 'basic', 'vision'
    }
    
    if skill_lower in generic_words:
        return True
    
    # Requirement phrases
    requirement_phrases = [
        'not required', 'if required', 'an asset', 'a plus',
        'preferred', 'required', 'optional', 'desirable'
    ]
    
    for phrase in requirement_phrases:
        if phrase in skill_lower:
            return True
    
    return False

def clean_skills_final_pass(input_file: str, output_file: str):
    """
    Final pass to remove remaining non-skills
    
    Args:
        input_file: Input CSV file
        output_file: Output CSV file
    """
    print("=" * 80)
    print("AI-ASSISTED SKILLS CLEANER - FINAL PASS")
    print("=" * 80)
    
    # Load data
    print(f"\n📂 Loading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df):,} skills")
    
    initial_count = len(df)
    
    # Remove explicit non-skills
    print(f"\n🗑️ Removing explicit non-skills...")
    df = df[~df['skill'].isin(NON_SKILLS_TO_REMOVE)]
    print(f"   Removed {initial_count - len(df):,} explicit non-skills")
    
    # Remove locations
    print(f"\n🌍 Removing locations (states, countries)...")
    before = len(df)
    df = df[~df['skill'].apply(is_location)]
    print(f"   Removed {before - len(df):,} locations")
    
    # Remove benefits/employment terms
    print(f"\n💼 Removing benefits and employment terms...")
    before = len(df)
    df = df[~df['skill'].apply(is_benefit_or_employment_term)]
    print(f"   Removed {before - len(df):,} benefits/employment terms")
    
    # Remove generic terms
    print(f"\n🔍 Removing generic/vague terms...")
    before = len(df)
    df = df[~df['skill'].apply(is_generic_term)]
    print(f"   Removed {before - len(df):,} generic terms")
    
    # Final cleanup: ensure no empty or very short entries
    print(f"\n🧹 Final cleanup...")
    before = len(df)
    df = df[df['skill'].str.len() >= 2]
    df = df[df['skill'].notna()]
    df = df[df['skill'] != '']
    print(f"   Removed {before - len(df):,} invalid entries")
    
    # Sort by frequency
    df = df.sort_values('frequency', ascending=False).reset_index(drop=True)
    
    # Save
    print(f"\n💾 Saving: {output_file}")
    df.to_csv(output_file, index=False)
    
    print(f"\n{'=' * 80}")
    print("✅ FINAL PASS COMPLETE")
    print(f"{'=' * 80}\n")
    
    print(f"📊 Summary:")
    print(f"   • Initial: {initial_count:,} skills")
    print(f"   • Final: {len(df):,} skills")
    print(f"   • Removed: {initial_count - len(df):,} non-skills ({(initial_count - len(df)) / initial_count * 100:.1f}%)")
    
    print(f"\n🔝 Top 30 final skills:")
    for i in range(min(30, len(df))):
        print(f"   {i+1:2d}. [{df.iloc[i]['frequency']:4d}x] {df.iloc[i]['skill']}")


if __name__ == "__main__":
    clean_skills_final_pass(
        input_file='data/skills_cleaned_v2.csv',
        output_file='data/skills_final.csv'
    )
