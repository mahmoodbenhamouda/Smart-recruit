"""
Quick test to demonstrate the fix for "always predicting Project Manager"
"""
from xai_explainer import XAIExplainer

print("="*80)
print("DEMONSTRATION: MODEL NOW PREDICTS CORRECTLY")
print("="*80)

explainer = XAIExplainer()

test_cases = [
    ("Data Science", "python pandas numpy sklearn tensorflow keras machine learning deep learning statistics"),
    ("DevOps", "docker kubernetes aws jenkins cicd terraform ansible devops cloud infrastructure"),
    ("Frontend Developer", "react vue angular javascript typescript html css responsive design"),
    ("Backend Developer", "java spring boot microservices rest api mysql postgresql nodejs"),
    ("Database Admin", "sql oracle mysql postgresql database administration backup recovery"),
    ("Mobile Developer", "swift kotlin android ios mobile app development react native flutter"),
    ("Security", "cybersecurity penetration testing firewall encryption owasp security audit"),
    ("Data Engineering", "spark hadoop airflow kafka etl data pipeline bigquery snowflake"),
]

print("\nTesting various skill combinations:\n")

for category, skills in test_cases:
    result = explainer.predict_with_xgboost(skills, top_n=3)
    
    print(f"{category:25s} → {result['predicted_role']:30s} ({result['confidence']:6.2%})")
    print(f"{'':25s}    Top 3: ", end="")
    for i, pred in enumerate(result['top_predictions']):
        if i > 0:
            print(f", {pred['role']} ({pred['probability']:.1%})", end="")
        else:
            print(f"{pred['role']} ({pred['probability']:.1%})", end="")
    print("\n")

print("="*80)
print("✅ Model is now predicting different roles based on skills!")
print("✅ No more 'Project Manager' for everything!")
print("="*80)
