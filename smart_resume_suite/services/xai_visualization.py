"""
Streamlit Visualization Components for Explainable AI
Provides interactive visualizations for SHAP and LIME explanations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional


def display_xai_explanation(explanation: Dict[str, Any]):
    """
    Display a comprehensive XAI explanation in Streamlit.
    
    Args:
        explanation: Dictionary containing XAI explanation data
    """
    if not explanation:
        st.warning("No XAI explanation available")
        return
    
    method = explanation.get('method', 'UNKNOWN')
    predicted_class = explanation.get('predicted_class', 'Unknown')
    confidence = explanation.get('confidence', 0)
    
    # Header
    st.markdown("### 🔍 Explainable AI Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Explanation Method", method)
    with col2:
        st.metric("Predicted Role", predicted_class)
    with col3:
        st.metric("Confidence", f"{confidence:.1%}")
    
    # Summary
    if 'summary' in explanation:
        st.info(f"**Summary:** {explanation['summary']}")
    
    # Display SHAP features
    if 'shap_features' in explanation and explanation['shap_features']:
        st.markdown("#### 📊 SHAP Feature Importance")
        display_shap_features(explanation['shap_features'])
    
    # Display LIME features
    if 'lime_features' in explanation and explanation['lime_features']:
        st.markdown("#### 📊 LIME Feature Weights")
        display_lime_features(explanation['lime_features'])
    
    # Comparison
    if 'comparison' in explanation and explanation['comparison']:
        st.markdown("#### ⚖️ Method Comparison")
        st.success(explanation['comparison'])


def display_shap_features(features: List[Dict[str, Any]], max_features: int = 15):
    """
    Display SHAP feature importance with visualization.
    
    Args:
        features: List of feature dictionaries with SHAP values
        max_features: Maximum number of features to display
    """
    if not features:
        st.warning("No SHAP features available")
        return
    
    # Prepare data
    features = features[:max_features]
    df = pd.DataFrame(features)
    
    # Sort by absolute SHAP value
    df['abs_shap'] = df['shap_value'].abs()
    df = df.sort_values('abs_shap', ascending=True)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Color based on positive/negative
    colors = ['#2ecc71' if impact == 'positive' else '#e74c3c' 
              for impact in df['impact']]
    
    fig.add_trace(go.Bar(
        y=df['feature'],
        x=df['shap_value'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{val:+.4f}" for val in df['shap_value']],
        textposition='outside',
        hovertemplate=(
            '<b>%{y}</b><br>' +
            'SHAP Value: %{x:.4f}<br>' +
            '<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title="SHAP Feature Importance",
        xaxis_title="SHAP Value (Impact on Prediction)",
        yaxis_title="Feature",
        height=max(400, len(df) * 30),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature table
    with st.expander("📋 View Detailed SHAP Values"):
        display_df = df[['feature', 'shap_value', 'impact', 'value']].copy()
        display_df['shap_value'] = display_df['shap_value'].round(4)
        display_df['value'] = display_df['value'].round(4)
        display_df.columns = ['Feature', 'SHAP Value', 'Impact', 'TF-IDF Value']
        st.dataframe(display_df.sort_values('SHAP Value', ascending=False, key=abs), use_container_width=True)


def display_lime_features(features: List[Dict[str, Any]], max_features: int = 15):
    """
    Display LIME feature weights with visualization.
    
    Args:
        features: List of feature dictionaries with LIME weights
        max_features: Maximum number of features to display
    """
    if not features:
        st.warning("No LIME features available")
        return
    
    # Prepare data
    features = features[:max_features]
    df = pd.DataFrame(features)
    
    # Sort by absolute weight
    df['abs_weight'] = df['lime_weight'].abs()
    df = df.sort_values('abs_weight', ascending=True)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    colors = ['#3498db' if impact == 'positive' else '#e67e22' 
              for impact in df['impact']]
    
    fig.add_trace(go.Bar(
        y=df['feature'],
        x=df['lime_weight'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{val:+.4f}" for val in df['lime_weight']],
        textposition='outside',
        hovertemplate=(
            '<b>%{y}</b><br>' +
            'LIME Weight: %{x:.4f}<br>' +
            '<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title="LIME Feature Weights",
        xaxis_title="LIME Weight (Feature Influence)",
        yaxis_title="Feature/Word",
        height=max(400, len(df) * 30),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature table
    with st.expander("📋 View Detailed LIME Weights"):
        display_df = df[['feature', 'lime_weight', 'impact']].copy()
        display_df['lime_weight'] = display_df['lime_weight'].round(4)
        display_df.columns = ['Feature/Word', 'LIME Weight', 'Impact']
        st.dataframe(display_df.sort_values('LIME Weight', ascending=False, key=abs), use_container_width=True)


def display_feature_comparison(shap_features: List[Dict], lime_features: List[Dict]):
    """
    Compare SHAP and LIME feature importances side-by-side.
    
    Args:
        shap_features: List of SHAP feature dictionaries
        lime_features: List of LIME feature dictionaries
    """
    st.markdown("#### ⚖️ SHAP vs LIME Feature Comparison")
    
    # Extract feature names
    shap_df = pd.DataFrame(shap_features[:10])
    lime_df = pd.DataFrame(lime_features[:10])
    
    # Get top features from both
    shap_top = set(shap_df['feature'].tolist())
    lime_top = set(f.split()[0] for f in lime_df['feature'].tolist())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top SHAP Features**")
        for feat in shap_df['feature'].head(10):
            impact = shap_df[shap_df['feature'] == feat]['impact'].iloc[0]
            icon = '✅' if impact == 'positive' else '❌'
            st.markdown(f"{icon} `{feat}`")
    
    with col2:
        st.markdown("**Top LIME Features**")
        for feat in lime_df['feature'].head(10):
            impact = lime_df[lime_df['feature'] == feat]['impact'].iloc[0]
            icon = '✅' if impact == 'positive' else '❌'
            st.markdown(f"{icon} `{feat}`")
    
    # Overlap metrics
    overlap = shap_top & lime_top
    st.markdown(f"**Agreement:** {len(overlap)} features appear in both top-10 lists")


def display_prediction_confidence(predictions: List[Dict[str, Any]]):
    """
    Display prediction confidence for top predictions.
    
    Args:
        predictions: List of prediction dictionaries with role and probability
    """
    if not predictions:
        return
    
    st.markdown("#### 🎯 Top Role Predictions")
    
    df = pd.DataFrame(predictions)
    
    # Bar chart
    fig = px.bar(
        df,
        x='probability',
        y='role',
        orientation='h',
        text=df['probability'].apply(lambda x: f"{x:.1%}"),
        color='probability',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title="Confidence",
        yaxis_title="Job Role",
        showlegend=False,
        height=max(300, len(df) * 60),
        xaxis=dict(tickformat='.0%')
    )
    
    fig.update_traces(textposition='outside')
    
    st.plotly_chart(fig, use_container_width=True)


def display_xai_insights(explanation: Dict[str, Any]):
    """
    Display actionable insights from XAI analysis.
    
    Args:
        explanation: XAI explanation dictionary
    """
    st.markdown("### 💡 Actionable Insights")
    
    shap_features = explanation.get('shap_features', [])
    lime_features = explanation.get('lime_features', [])
    
    # Positive features (strengths)
    positive_features = []
    if shap_features:
        positive_features = [f['feature'] for f in shap_features if f['impact'] == 'positive'][:5]
    
    # Negative features (areas to improve)
    negative_features = []
    if shap_features:
        negative_features = [f['feature'] for f in shap_features if f['impact'] == 'negative'][:5]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("**🌟 Your Strengths**")
        if positive_features:
            for feat in positive_features:
                st.markdown(f"- **{feat}**: Strong positive indicator")
        else:
            st.markdown("_No clear strengths identified_")
    
    with col2:
        st.warning("**📈 Areas to Improve**")
        if negative_features:
            for feat in negative_features:
                st.markdown(f"- **{feat}**: Consider developing this area")
        else:
            st.markdown("_No major gaps identified_")
    
    # Recommendations
    st.markdown("### 🎯 Recommendations")
    
    predicted_class = explanation.get('predicted_class', 'Unknown')
    confidence = explanation.get('confidence', 0)
    
    if confidence < 0.5:
        st.info(
            f"Low confidence ({confidence:.1%}) for **{predicted_class}**. "
            "Consider enhancing skills mentioned in negative features."
        )
    elif confidence < 0.7:
        st.success(
            f"Moderate confidence ({confidence:.1%}) for **{predicted_class}**. "
            "You're on the right track! Focus on strengthening your core skills."
        )
    else:
        st.success(
            f"High confidence ({confidence:.1%}) for **{predicted_class}**! "
            "Your profile strongly matches this role."
        )


def create_xai_dashboard(analysis_result: Dict[str, Any]):
    """
    Create a complete XAI dashboard for resume analysis.
    
    Args:
        analysis_result: Complete analysis result with XAI explanations
    """
    st.title("🤖 Explainable AI Resume Analysis")
    
    # Basic info
    if 'xai_explanation' in analysis_result and analysis_result['xai_explanation']:
        xai_exp = analysis_result['xai_explanation']
        
        # Main explanation display
        display_xai_explanation(xai_exp)
        
        st.markdown("---")
        
        # Insights
        display_xai_insights(xai_exp)
        
        st.markdown("---")
        
        # Job prediction
        if 'job_prediction' in analysis_result and analysis_result['job_prediction']:
            job_pred = analysis_result['job_prediction']
            if 'top_predictions' in job_pred:
                display_prediction_confidence(job_pred['top_predictions'])
    else:
        st.warning("XAI explanation not available in the analysis result")
        
        # Show standard analysis if available
        if 'matched_skills' in analysis_result:
            st.markdown("### Matched Skills")
            skills = analysis_result['matched_skills']
            if skills:
                cols = st.columns(3)
                for i, skill in enumerate(skills[:15]):
                    cols[i % 3].markdown(f"✅ {skill}")
        
        if 'missing_skills' in analysis_result:
            st.markdown("### Missing Skills")
            skills = analysis_result['missing_skills']
            if skills:
                cols = st.columns(3)
                for i, skill in enumerate(skills[:15]):
                    cols[i % 3].markdown(f"❌ {skill}")


# Example usage
def main():
    """Example Streamlit app with XAI visualizations"""
    st.set_page_config(page_title="XAI Resume Analyzer", layout="wide")
    
    st.title("🤖 Explainable AI Resume Analyzer")
    st.markdown("Upload a resume to see SHAP and LIME explanations")
    
    # This is a demo - in production, connect to your API
    st.info("👆 This is a visualization component. Integrate with your ATS API to get real data.")
    
    # Demo data
    demo_explanation = {
        'method': 'BOTH',
        'predicted_class': 'Machine Learning Engineer',
        'confidence': 0.85,
        'summary': 'Predicted role: Machine Learning Engineer (confidence: 85%). Top SHAP feature: python (positive). Top LIME feature: tensorflow (positive)',
        'shap_features': [
            {'feature': 'python', 'value': 0.8, 'shap_value': 0.25, 'impact': 'positive'},
            {'feature': 'tensorflow', 'value': 0.6, 'shap_value': 0.20, 'impact': 'positive'},
            {'feature': 'machine learning', 'value': 0.7, 'shap_value': 0.18, 'impact': 'positive'},
            {'feature': 'java', 'value': 0.1, 'shap_value': -0.05, 'impact': 'negative'},
        ],
        'lime_features': [
            {'feature': 'tensorflow', 'lime_weight': 0.22, 'impact': 'positive'},
            {'feature': 'python', 'lime_weight': 0.20, 'impact': 'positive'},
            {'feature': 'neural', 'lime_weight': 0.15, 'impact': 'positive'},
        ],
        'comparison': 'SHAP and LIME agree on 2 features. High agreement indicates robust explanation.'
    }
    
    if st.button("Show Demo Explanation"):
        display_xai_explanation(demo_explanation)
        st.markdown("---")
        display_xai_insights(demo_explanation)


if __name__ == "__main__":
    main()
