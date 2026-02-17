"""
Debug script to check model predictions
"""
import pickle
import numpy as np

print("Loading model components...")
model = pickle.load(open('JobPrediction_Model/xgboost_model.pkl', 'rb'))
vectorizer = pickle.load(open('JobPrediction_Model/xgb_tfidf_vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open('JobPrediction_Model/xgb_label_encoder.pkl', 'rb'))

print(f"\nModel info:")
print(f"  Type: {type(model)}")
print(f"  N classes: {model.n_classes_}")
print(f"  Classes: {label_encoder.classes_[:10]}... (showing first 10)")

print(f"\nVectorizer info:")
print(f"  Max features: {vectorizer.max_features}")
print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

# Test with different inputs
test_inputs = [
    "python machine learning data science tensorflow",
    "devops docker kubernetes aws",
    "react javascript frontend web development",
    "sql database management oracle"
]

print("\n" + "="*80)
print("TESTING PREDICTIONS")
print("="*80)

for test_input in test_inputs:
    print(f"\nInput: {test_input}")
    
    # Transform input
    X = vectorizer.transform([test_input])
    print(f"  Transformed shape: {X.shape}")
    print(f"  Non-zero features: {X.nnz}")
    
    # Get prediction
    prediction = model.predict(X)[0]
    predicted_role = label_encoder.inverse_transform([prediction])[0]
    
    # Get probabilities
    proba = model.predict_proba(X)[0]
    max_proba = proba[prediction]
    
    print(f"  Predicted class index: {prediction}")
    print(f"  Predicted role: {predicted_role}")
    print(f"  Confidence: {max_proba:.2%}")
    
    # Show top 3 predictions
    top_3_indices = np.argsort(proba)[-3:][::-1]
    print(f"  Top 3:")
    for idx in top_3_indices:
        role = label_encoder.inverse_transform([idx])[0]
        print(f"    - {role}: {proba[idx]:.2%}")

print("\n" + "="*80)
print("CHECKING MODEL WEIGHTS")
print("="*80)

# Check if model has been properly trained
print(f"\nModel estimators: {model.n_estimators}")
print(f"Model max depth: {model.max_depth}")

# Check feature importance
if hasattr(model, 'feature_importances_'):
    importance = model.feature_importances_
    print(f"\nFeature importances shape: {importance.shape}")
    print(f"Non-zero importances: {np.count_nonzero(importance)}")
    print(f"Max importance: {importance.max():.6f}")
    print(f"Min importance: {importance.min():.6f}")
    
    # Get top features
    top_feature_indices = np.argsort(importance)[-10:][::-1]
    feature_names = vectorizer.get_feature_names_out()
    print(f"\nTop 10 most important features:")
    for i, idx in enumerate(top_feature_indices, 1):
        print(f"  {i}. {feature_names[idx]}: {importance[idx]:.6f}")

# Check class distribution in predictions
print("\n" + "="*80)
print("CHECKING CLASS PREDICTIONS ON RANDOM SAMPLES")
print("="*80)

# Generate some test cases
test_cases = [
    "python programming software development",
    "java spring boot backend",
    "javascript react nodejs",
    "aws cloud devops",
    "machine learning artificial intelligence",
    "data analysis statistics",
    "project management agile scrum",
    "sales marketing customer",
    "accounting finance excel",
    "design ui ux figma"
]

predictions_count = {}
for test in test_cases:
    X = vectorizer.transform([test])
    pred = model.predict(X)[0]
    role = label_encoder.inverse_transform([pred])[0]
    predictions_count[role] = predictions_count.get(role, 0) + 1

print(f"\nPrediction distribution across {len(test_cases)} test cases:")
for role, count in sorted(predictions_count.items(), key=lambda x: x[1], reverse=True):
    print(f"  {role}: {count} ({count/len(test_cases)*100:.0f}%)")

if len(predictions_count) == 1:
    print("\n❌ WARNING: Model is predicting the same class for all inputs!")
    print("   This indicates the model is not properly trained.")
else:
    print(f"\n✅ Model predicts {len(predictions_count)} different classes")
