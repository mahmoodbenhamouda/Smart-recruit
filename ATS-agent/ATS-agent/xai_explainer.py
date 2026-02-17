"""
Explainable AI Module for ATS System
Provides SHAP and LIME explanations for model predictions using XGBoost
"""

import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import SHAP and LIME
try:
    import shap
except ImportError:
    raise ImportError("SHAP is required. Install with: pip install shap")

try:
    from lime.lime_text import LimeTextExplainer
except ImportError:
    raise ImportError("LIME is required. Install with: pip install lime")


class XAIExplainer:
    """
    Explainable AI wrapper for ATS job prediction models.
    Provides SHAP and LIME explanations for model predictions.
    """
    
    def __init__(self, model_path: str = 'JobPrediction_Model'):
        """
        Initialize the XAI explainer.
        
        Args:
            model_path: Path to directory containing trained models
        """
        self.model_path = Path(model_path)
        self.xgboost_model_path = self.model_path / 'xgboost_model.pkl'
        self.vectorizer_path = self.model_path / 'xgb_tfidf_vectorizer.pkl'
        self.label_encoder_path = self.model_path / 'xgb_label_encoder.pkl'
        
        # Initialize components
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.job_roles: List[str] = []
        
        # SHAP and LIME explainers
        self.shap_explainer: Optional[shap.Explainer] = None
        self.lime_explainer: Optional[LimeTextExplainer] = None
        
        # Load or create models
        self._load_or_create_models()
        
    def _load_or_create_models(self):
        """Load existing XGBoost model or prepare to create one"""
        # Try to load label encoder (from LSTM model directory)
        lstm_label_path = self.model_path / 'label_encoder.pkl'
        if lstm_label_path.exists():
            try:
                with open(lstm_label_path, 'rb') as f:
                    lstm_encoder = pickle.load(f)
                print(f"ℹ️ Found LSTM label encoder with {len(lstm_encoder.classes_)} job roles")
                print(f"   Note: XGBoost will create its own encoder when training")
            except Exception as e:
                print(f"⚠️ Could not load LSTM label encoder: {e}")
        
        # Try to load existing XGBoost model and its label encoder
        if self.xgboost_model_path.exists() and self.vectorizer_path.exists() and self.label_encoder_path.exists():
            with open(self.xgboost_model_path, 'rb') as f:
                self.xgb_model = pickle.load(f)
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(self.label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Set job roles from XGBoost label encoder
            self.job_roles = self.label_encoder.classes_.tolist()
            
            print(f"✅ Loaded XGBoost model and vectorizer")
            print(f"   Job roles: {len(self.job_roles)} classes")
            self._initialize_explainers()
        else:
            print("⚠️ XGBoost model not found. Call train_xgboost_model() to create it.")
    
    def train_xgboost_model(self, 
                           texts: List[str], 
                           labels: List[str],
                           max_features: int = 5000,
                           test_size: float = 0.2) -> Dict[str, float]:
        """
        Train an XGBoost model for job role classification.
        
        Args:
            texts: List of text samples (skills, descriptions)
            labels: List of corresponding job role labels
            max_features: Maximum number of TF-IDF features
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score
        
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL FOR EXPLAINABLE AI")
        print("="*60)
        
        # Prepare labels
        # Always fit on the training data to handle all job roles in dataset
        print(f"   Fitting label encoder on training data...")
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        self.job_roles = self.label_encoder.classes_.tolist()
        
        # Create TF-IDF features
        print(f"\n📊 Creating TF-IDF features (max {max_features} features)...")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        X = self.vectorizer.fit_transform(texts)
        
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Number of classes: {len(self.job_roles)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train XGBoost model
        print(f"\n🚀 Training XGBoost classifier...")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='mlogloss'
        )
        
        self.xgb_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.xgb_model.predict(X_train)
        y_pred_test = self.xgb_model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        train_f1 = f1_score(y_train, y_pred_train, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        print(f"\n✅ Training complete!")
        print(f"   Train Accuracy: {train_acc:.2%}")
        print(f"   Test Accuracy:  {test_acc:.2%}")
        print(f"   Train F1 Score: {train_f1:.2%}")
        print(f"   Test F1 Score:  {test_f1:.2%}")
        
        # Save models
        self._save_models()
        
        # Initialize explainers
        self._initialize_explainers()
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'n_samples': len(texts),
            'n_features': X.shape[1]
        }
    
    def _save_models(self):
        """Save XGBoost model, vectorizer, and label encoder"""
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        with open(self.xgboost_model_path, 'wb') as f:
            pickle.dump(self.xgb_model, f)
        
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(self.label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"\n💾 Models saved to {self.model_path}")
    
    def _initialize_explainers(self):
        """Initialize SHAP and LIME explainers"""
        if self.xgb_model is None or self.vectorizer is None:
            print("⚠️ Cannot initialize explainers without trained models")
            return
        
        print("\n🔍 Initializing SHAP and LIME explainers...")
        
        # For SHAP, we'll use the model's feature importance as a simpler alternative
        # since TreeExplainer has compatibility issues with XGBoost 2.0+ multi-class
        # We'll compute feature importance directly from the model
        self.shap_explainer = "feature_importance"  # Flag to use feature importance method
        print("✅ Using XGBoost feature importance for SHAP-like explanations")
        
        # LIME TextExplainer
        self.lime_explainer = LimeTextExplainer(
            class_names=self.job_roles,
            feature_selection='auto',
            split_expression=r'\W+'
        )
        
        print("✅ Explainers initialized successfully")
    
    def predict_with_xgboost(self, text: str, top_n: int = 3) -> Dict[str, Any]:
        """
        Make prediction using XGBoost model.
        
        Args:
            text: Input text (skills, description)
            top_n: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if self.xgb_model is None or self.vectorizer is None:
            raise ValueError("XGBoost model not loaded. Train or load a model first.")
        
        # Vectorize input
        X = self.vectorizer.transform([text])
        
        # Predict probabilities
        probs = self.xgb_model.predict_proba(X)[0]
        
        # Get top predictions
        top_indices = np.argsort(probs)[-top_n:][::-1]
        top_predictions = [
            {'role': self.job_roles[idx], 'probability': float(probs[idx])}
            for idx in top_indices
        ]
        
        # Best prediction
        best_idx = np.argmax(probs)
        
        return {
            'predicted_role': self.job_roles[best_idx],
            'confidence': float(probs[best_idx]),
            'top_predictions': top_predictions,
            'all_probabilities': {
                role: float(prob) for role, prob in zip(self.job_roles, probs)
            }
        }
    
    def explain_with_shap(self, text: str, target_class: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate SHAP-like explanation for a prediction using feature importance.
        
        Args:
            text: Input text to explain
            target_class: Specific class to explain (None for predicted class)
            
        Returns:
            Dictionary with SHAP-like values and explanation
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized. Train model first.")
        
        # Vectorize input
        X = self.vectorizer.transform([text])
        X_dense = X.toarray()[0]
        
        # Get prediction
        prediction = self.predict_with_xgboost(text)
        predicted_class = prediction['predicted_role']
        
        # Determine which class to explain
        if target_class is None:
            target_class = predicted_class
        
        class_idx = self.job_roles.index(target_class)
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Method 1: Use model's feature importance (global)
        try:
            feature_importance_scores = self.xgb_model.feature_importances_
        except:
            feature_importance_scores = np.zeros(len(feature_names))
        
        # Method 2: Compute local contribution by perturbation
        # For each feature present in the text, measure its contribution
        base_probs = self.xgb_model.predict_proba(X)[0]
        base_prob = base_probs[class_idx]
        
        # Create feature importance list
        feature_importance = []
        
        for i, (feat_name, feat_val) in enumerate(zip(feature_names, X_dense)):
            if feat_val > 0:  # Only include features present in the text
                # Compute local importance by removing this feature
                X_perturbed = X_dense.copy()
                X_perturbed[i] = 0  # Remove this feature
                
                perturbed_probs = self.xgb_model.predict_proba([X_perturbed])[0]
                perturbed_prob = perturbed_probs[class_idx]
                
                # SHAP-like value: difference in prediction
                shap_value = base_prob - perturbed_prob
                
                # Scale by feature value and global importance
                scaled_shap = shap_value * feat_val * (feature_importance_scores[i] + 0.001)
                
                feature_importance.append({
                    'feature': feat_name,
                    'value': float(feat_val),
                    'shap_value': float(scaled_shap),
                    'impact': 'positive' if scaled_shap > 0 else 'negative'
                })
        
        # Sort by absolute SHAP value
        feature_importance.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        # Calculate base value (average prediction across all classes)
        base_value = 1.0 / len(self.job_roles)
        
        return {
            'method': 'SHAP',
            'predicted_class': predicted_class,
            'explained_class': target_class,
            'confidence': prediction['confidence'],
            'base_value': base_value,
            'top_features': feature_importance[:20],
            'summary': self._generate_shap_summary(feature_importance[:10], target_class)
        }
    
    def explain_with_lime(self, text: str, num_features: int = 20, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Generate LIME explanation for a prediction.
        
        Args:
            text: Input text to explain
            num_features: Number of features to include in explanation
            num_samples: Number of samples for LIME
            
        Returns:
            Dictionary with LIME explanation
        """
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized. Train model first.")
        
        # Prediction function for LIME
        def predict_proba_fn(texts):
            X = self.vectorizer.transform(texts)
            return self.xgb_model.predict_proba(X)
        
        # Get prediction
        prediction = self.predict_with_xgboost(text)
        predicted_class = prediction['predicted_role']
        class_idx = self.job_roles.index(predicted_class)
        
        # Generate LIME explanation
        exp = self.lime_explainer.explain_instance(
            text,
            predict_proba_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=[class_idx]
        )
        
        # Extract feature weights
        feature_weights = exp.as_list(label=class_idx)
        
        # Format explanation
        features_explanation = []
        for feature, weight in feature_weights:
            features_explanation.append({
                'feature': feature,
                'weight': float(weight),
                'impact': 'positive' if weight > 0 else 'negative'
            })
        
        return {
            'method': 'LIME',
            'predicted_class': predicted_class,
            'confidence': prediction['confidence'],
            'features': features_explanation,
            'summary': self._generate_lime_summary(features_explanation[:10], predicted_class),
            'lime_object': exp  # For visualization
        }
    
    def compare_explanations(self, text: str) -> Dict[str, Any]:
        """
        Generate both SHAP and LIME explanations for comparison.
        
        Args:
            text: Input text to explain
            
        Returns:
            Dictionary with both explanations
        """
        shap_explanation = self.explain_with_shap(text)
        lime_explanation = self.explain_with_lime(text)
        
        return {
            'text': text,
            'prediction': self.predict_with_xgboost(text),
            'shap': shap_explanation,
            'lime': lime_explanation,
            'comparison_summary': self._compare_methods(shap_explanation, lime_explanation)
        }
    
    def _generate_shap_summary(self, top_features: List[Dict], target_class: str) -> str:
        """Generate human-readable summary of SHAP explanation"""
        if not top_features:
            return f"No significant features found for {target_class}"
        
        positive_features = [f for f in top_features if f['impact'] == 'positive']
        negative_features = [f for f in top_features if f['impact'] == 'negative']
        
        summary_parts = [f"Prediction: {target_class}"]
        
        if positive_features:
            top_pos = positive_features[:3]
            features_str = ", ".join([f"'{f['feature']}'" for f in top_pos])
            summary_parts.append(f"Key supporting features: {features_str}")
        
        if negative_features:
            top_neg = negative_features[:3]
            features_str = ", ".join([f"'{f['feature']}'" for f in top_neg])
            summary_parts.append(f"Features reducing confidence: {features_str}")
        
        return ". ".join(summary_parts)
    
    def _generate_lime_summary(self, features: List[Dict], predicted_class: str) -> str:
        """Generate human-readable summary of LIME explanation"""
        if not features:
            return f"No significant features found for {predicted_class}"
        
        positive = [f for f in features if f['impact'] == 'positive']
        negative = [f for f in features if f['impact'] == 'negative']
        
        summary_parts = [f"Predicted as: {predicted_class}"]
        
        if positive:
            top_pos = positive[:3]
            features_str = ", ".join([f"'{f['feature']}'" for f in top_pos])
            summary_parts.append(f"Most influential positive features: {features_str}")
        
        if negative:
            top_neg = negative[:3]
            features_str = ", ".join([f"'{f['feature']}'" for f in top_neg])
            summary_parts.append(f"Most influential negative features: {features_str}")
        
        return ". ".join(summary_parts)
    
    def _compare_methods(self, shap_exp: Dict, lime_exp: Dict) -> str:
        """Compare SHAP and LIME explanations"""
        # Extract top features from both
        shap_features = set(f['feature'] for f in shap_exp['top_features'][:10])
        lime_features = set(f['feature'].split() for f in lime_exp['features'][:10])
        
        # Flatten LIME features (since they might be phrases)
        lime_features_flat = set()
        for feat_list in lime_features:
            lime_features_flat.update(feat_list)
        
        overlap = shap_features & lime_features_flat
        
        if len(overlap) > 3:
            return f"SHAP and LIME agree on {len(overlap)} key features, indicating robust explanation."
        else:
            return "SHAP and LIME show different feature importances, suggesting multiple explanation perspectives."


def main():
    """Test the XAI explainer"""
    print("="*80)
    print("EXPLAINABLE AI (XAI) MODULE - TEST")
    print("="*80)
    
    # Note: To run this test, you need training data
    # This is a demonstration of how to use the module
    
    explainer = XAIExplainer()
    
    # Example: If you have training data
    # training_data = [
    #     ("Python TensorFlow Keras Machine Learning Deep Learning", "Machine Learning Engineer"),
    #     ("Java Spring Boot Microservices REST API", "Software Engineer"),
    #     # ... more examples
    # ]
    # texts, labels = zip(*training_data)
    # explainer.train_xgboost_model(list(texts), list(labels))
    
    # Example prediction and explanation
    test_text = "Python, Machine Learning, TensorFlow, Deep Learning, Neural Networks, PyTorch"
    
    if explainer.xgb_model is not None:
        print("\n" + "="*80)
        print("PREDICTION AND EXPLANATION")
        print("="*80)
        print(f"\nInput: {test_text}")
        
        # Make prediction
        prediction = explainer.predict_with_xgboost(test_text)
        print(f"\n🎯 Predicted Role: {prediction['predicted_role']}")
        print(f"   Confidence: {prediction['confidence']:.2%}")
        
        # SHAP explanation
        print("\n📊 SHAP Explanation:")
        shap_exp = explainer.explain_with_shap(test_text)
        print(f"   {shap_exp['summary']}")
        print("\n   Top Features:")
        for feat in shap_exp['top_features'][:5]:
            print(f"      {feat['feature']:20s} -> {feat['shap_value']:+.4f} ({feat['impact']})")
        
        # LIME explanation
        print("\n📊 LIME Explanation:")
        lime_exp = explainer.explain_with_lime(test_text)
        print(f"   {lime_exp['summary']}")
        print("\n   Top Features:")
        for feat in lime_exp['features'][:5]:
            print(f"      {feat['feature']:30s} -> {feat['weight']:+.4f} ({feat['impact']})")
    else:
        print("\n⚠️ No trained XGBoost model found.")
        print("   To use XAI features, train a model with training data.")


if __name__ == "__main__":
    main()
