"""
Streamlit App to Test XAI Features
Interactive testing of SHAP and LIME explanations
"""

import streamlit as st
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from xai_explainer import XAIExplainer

# Page config
st.set_page_config(
    page_title="XAI Testing Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .feature-positive {
        color: #2ecc71;
        font-weight: bold;
    }
    .feature-negative {
        color: #e74c3c;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'explainer' not in st.session_state:
    with st.spinner("🔄 Loading XAI Explainer..."):
        try:
            # Get the directory where this script is located
            script_dir = Path(__file__).parent
            model_path = script_dir / 'JobPrediction_Model'
            
            st.session_state.explainer = XAIExplainer(model_path=str(model_path))
            st.session_state.model_loaded = (
                st.session_state.explainer.xgb_model is not None
            )
        except Exception as e:
            st.error(f"Failed to load explainer: {e}")
            st.session_state.model_loaded = False

# Header
st.markdown('<div class="main-header">🤖 XAI Testing Dashboard</div>', unsafe_allow_html=True)
st.markdown("Test SHAP and LIME explanations for job role predictions")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model status
    if st.session_state.model_loaded:
        st.success("✅ XGBoost Model Loaded")
        st.info(f"📊 {len(st.session_state.explainer.job_roles)} Job Roles")
    else:
        st.error("❌ Model Not Loaded")
        st.warning("Run: `python train_xai_model.py`")
        st.stop()
    
    st.markdown("---")
    
    # Explanation settings
    st.subheader("Explanation Settings")
    
    num_top_features = st.slider(
        "Top Features to Show",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )
    
    lime_samples = st.slider(
        "LIME Samples",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="More samples = more accurate but slower"
    )
    
    show_all_predictions = st.checkbox(
        "Show All Predictions",
        value=False,
        help="Display all job role probabilities"
    )
    
    st.markdown("---")
    
    # Example templates
    st.subheader("📝 Example Skills")
    if st.button("ML Engineer"):
        st.session_state.skills_input = "Python TensorFlow Keras PyTorch Deep Learning Neural Networks Machine Learning Computer Vision NLP"
    
    if st.button("DevOps Engineer"):
        st.session_state.skills_input = "Docker Kubernetes AWS Jenkins CI/CD Terraform Ansible Linux DevOps Cloud Infrastructure Monitoring"
    
    if st.button("Data Scientist"):
        st.session_state.skills_input = "Python R SQL Pandas NumPy Statistics Machine Learning Data Analysis Visualization Jupyter Matplotlib"
    
    if st.button("Software Engineer"):
        st.session_state.skills_input = "Java Python JavaScript React Node.js Spring Boot REST API Git Agile Microservices Database"
    
    if st.button("Data Engineer"):
        st.session_state.skills_input = "Python SQL Spark Hadoop ETL Data Pipeline Airflow Kafka Snowflake Big Data Redshift"

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predict & Explain", "📊 Compare Methods", "🔍 Explore Job Roles", "📈 Batch Analysis"])

# Tab 1: Predict & Explain
with tab1:
    st.markdown('<div class="sub-header">🎯 Job Role Prediction with Explanations</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input
        skills_input = st.text_area(
            "Enter Skills (comma or space separated)",
            value=st.session_state.get('skills_input', ''),
            height=150,
            placeholder="e.g., Python, TensorFlow, Machine Learning, Deep Learning...",
            key="skills_text_tab1"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            predict_btn = st.button("🎯 Predict Role", type="primary", use_container_width=True)
        with col_btn2:
            explain_shap_btn = st.button("📊 SHAP Explanation", use_container_width=True)
        with col_btn3:
            explain_lime_btn = st.button("📊 LIME Explanation", use_container_width=True)
    
    with col2:
        st.info("**Tips:**\n- Use the sidebar for quick examples\n- Add multiple skills for better predictions\n- Try different combinations")
    
    if predict_btn and skills_input.strip():
        with st.spinner("🔄 Predicting..."):
            try:
                result = st.session_state.explainer.predict_with_xgboost(
                    skills_input,
                    top_n=5
                )
                
                # Display results
                st.markdown("---")
                st.markdown("### 🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Role", result['predicted_role'])
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                with col3:
                    conf_level = "High" if result['confidence'] > 0.7 else "Medium" if result['confidence'] > 0.4 else "Low"
                    st.metric("Confidence Level", conf_level)
                
                # Top predictions
                st.markdown("#### 📊 Top 5 Predictions")
                for i, pred in enumerate(result['top_predictions'], 1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{i}. {pred['role']}**")
                    with col2:
                        st.progress(pred['probability'])
                        st.write(f"{pred['probability']:.1%}")
                
                # All predictions (if enabled)
                if show_all_predictions:
                    with st.expander("📋 All Job Role Probabilities"):
                        import pandas as pd
                        df = pd.DataFrame([
                            {'Job Role': role, 'Probability': prob}
                            for role, prob in sorted(
                                result['all_probabilities'].items(),
                                key=lambda x: x[1],
                                reverse=True
                            )
                        ])
                        st.dataframe(df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
    
    if explain_shap_btn and skills_input.strip():
        with st.spinner("🔄 Generating SHAP explanation... (may take ~2 seconds)"):
            try:
                shap_result = st.session_state.explainer.explain_with_shap(skills_input)
                
                st.markdown("---")
                st.markdown("### 📊 SHAP Explanation")
                
                # Summary
                st.info(f"**Summary:** {shap_result['summary']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Class", shap_result['predicted_class'])
                with col2:
                    st.metric("Confidence", f"{shap_result['confidence']:.1%}")
                with col3:
                    st.metric("Base Value", f"{shap_result['base_value']:.4f}")
                
                # Feature importance
                st.markdown("#### 🔍 Feature Importance (SHAP Values)")
                
                st.info("💡 **How to read SHAP values:** These show how much each feature contributes to the prediction probability. Larger absolute values mean stronger influence.")
                
                features = shap_result['top_features'][:num_top_features]
                if features:
                    import pandas as pd
                    
                    df = pd.DataFrame(features)
                    df['abs_shap'] = df['shap_value'].abs()
                    df = df.sort_values('abs_shap', ascending=False)
                    
                    # Scale SHAP values for better visualization (multiply by 100 for percentage points)
                    df['shap_scaled'] = df['shap_value'] * 100
                    
                    # Bar chart
                    import plotly.graph_objects as go
                    
                    colors = ['#2ecc71' if imp == 'positive' else '#e74c3c' for imp in df['impact']]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=df['shap_scaled'],
                            y=df['feature'],
                            orientation='h',
                            marker=dict(color=colors),
                            text=[f"{val:+.3f}pp" for val in df['shap_scaled']],
                            textposition='outside',
                            hovertemplate=(
                                '<b>%{y}</b><br>' +
                                'Contribution: %{x:.4f} percentage points<br>' +
                                '<extra></extra>'
                            )
                        )
                    ])
                    
                    fig.update_layout(
                        title="SHAP Feature Contributions (in percentage points)",
                        xaxis_title="Contribution to Prediction Probability (percentage points)",
                        yaxis_title="Feature",
                        height=max(400, len(df) * 30),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show table
                    st.markdown("##### Top Contributing Features:")
                    display_df = df[['feature', 'shap_scaled', 'impact']].copy()
                    display_df.columns = ['Feature', 'Contribution (pp)', 'Impact']
                    display_df['Contribution (pp)'] = display_df['Contribution (pp)'].apply(lambda x: f"{x:+.3f}")
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    st.caption("**pp** = percentage points. E.g., +0.250pp means this feature increases the probability by 0.25%")
                    
                    # Table
                    with st.expander("📋 Detailed SHAP Values (raw)"):
                        st.dataframe(
                            df[['feature', 'shap_value', 'impact', 'value']].round(6),
                            use_container_width=True
                        )
                
            except Exception as e:
                st.error(f"❌ SHAP explanation failed: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    if explain_lime_btn and skills_input.strip():
        with st.spinner("🔄 Generating LIME explanation... (may take ~1 second)"):
            try:
                lime_result = st.session_state.explainer.explain_with_lime(
                    skills_input,
                    num_features=num_top_features,
                    num_samples=lime_samples
                )
                
                st.markdown("---")
                st.markdown("### 📊 LIME Explanation")
                
                # Summary
                st.info(f"**Summary:** {lime_result['summary']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Class", lime_result['predicted_class'])
                with col2:
                    st.metric("Confidence", f"{lime_result['confidence']:.1%}")
                
                # Feature weights
                st.markdown("#### 🔍 Feature Weights")
                
                features = lime_result['features'][:num_top_features]
                if features:
                    import pandas as pd
                    import plotly.graph_objects as go
                    
                    df = pd.DataFrame(features)
                    df['abs_weight'] = df['lime_weight'].abs()
                    df = df.sort_values('abs_weight', ascending=False)
                    
                    # Bar chart
                    colors = ['#3498db' if imp == 'positive' else '#e67e22' for imp in df['impact']]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=df['lime_weight'],
                            y=df['feature'],
                            orientation='h',
                            marker=dict(color=colors),
                            text=[f"{val:+.4f}" for val in df['lime_weight']],
                            textposition='outside'
                        )
                    ])
                    
                    fig.update_layout(
                        title="LIME Feature Weights",
                        xaxis_title="LIME Weight",
                        yaxis_title="Feature/Word",
                        height=max(400, len(df) * 30),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Table
                    with st.expander("📋 Detailed LIME Weights"):
                        st.dataframe(
                            df[['feature', 'lime_weight', 'impact']].round(4),
                            use_container_width=True
                        )
                
            except Exception as e:
                st.error(f"❌ LIME explanation failed: {e}")

# Tab 2: Compare Methods
with tab2:
    st.markdown('<div class="sub-header">📊 Compare SHAP vs LIME</div>', unsafe_allow_html=True)
    
    skills_input_compare = st.text_area(
        "Enter Skills for Comparison",
        height=100,
        placeholder="e.g., Docker, Kubernetes, AWS, DevOps...",
        key="skills_compare"
    )
    
    if st.button("🔍 Compare Methods", type="primary"):
        if skills_input_compare.strip():
            with st.spinner("🔄 Generating both explanations..."):
                try:
                    comparison = st.session_state.explainer.compare_explanations(skills_input_compare)
                    
                    st.markdown("---")
                    
                    # Prediction
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Role", comparison['prediction']['predicted_role'])
                    with col2:
                        st.metric("Confidence", f"{comparison['prediction']['confidence']:.1%}")
                    
                    # Comparison summary
                    st.info(f"**Comparison:** {comparison['comparison_summary']}")
                    
                    # Side by side
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📊 SHAP Analysis")
                        st.write(comparison['shap']['summary'])
                        
                        shap_features = comparison['shap']['top_features'][:10]
                        for feat in shap_features:
                            impact_icon = '✅' if feat['impact'] == 'positive' else '❌'
                            st.markdown(f"{impact_icon} **{feat['feature']}**: {feat['shap_value']:+.4f}")
                    
                    with col2:
                        st.markdown("#### 📊 LIME Analysis")
                        st.write(comparison['lime']['summary'])
                        
                        lime_features = comparison['lime']['features'][:10]
                        for feat in lime_features:
                            impact_icon = '✅' if feat['impact'] == 'positive' else '❌'
                            st.markdown(f"{impact_icon} **{feat['feature']}**: {feat['weight']:+.4f}")
                    
                except Exception as e:
                    st.error(f"❌ Comparison failed: {e}")
        else:
            st.warning("Please enter some skills to compare")

# Tab 3: Explore Job Roles
with tab3:
    st.markdown('<div class="sub-header">🔍 Explore Available Job Roles</div>', unsafe_allow_html=True)
    
    job_roles = st.session_state.explainer.job_roles
    
    st.write(f"**Total Job Roles:** {len(job_roles)}")
    
    # Display in columns
    cols_per_row = 3
    for i in range(0, len(job_roles), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(job_roles):
                with col:
                    st.info(f"**{i+j+1}.** {job_roles[i+j]}")
    
    # Search
    st.markdown("---")
    search_term = st.text_input("🔍 Search Job Roles", placeholder="e.g., engineer, data, machine...")
    
    if search_term:
        matching = [role for role in job_roles if search_term.lower() in role.lower()]
        if matching:
            st.success(f"Found {len(matching)} matching roles:")
            for role in matching:
                st.write(f"• {role}")
        else:
            st.warning("No matching roles found")

# Tab 4: Batch Analysis
with tab4:
    st.markdown('<div class="sub-header">📈 Batch Analysis</div>', unsafe_allow_html=True)
    
    st.write("Analyze multiple skill sets at once")
    
    batch_input = st.text_area(
        "Enter Multiple Skill Sets (one per line)",
        height=200,
        placeholder="Python TensorFlow Machine Learning\nDocker Kubernetes AWS DevOps\nJava Spring Boot REST API",
        key="batch_input"
    )
    
    if st.button("🚀 Analyze Batch", type="primary"):
        if batch_input.strip():
            lines = [line.strip() for line in batch_input.split('\n') if line.strip()]
            
            if lines:
                results = []
                progress_bar = st.progress(0)
                
                for i, skills in enumerate(lines):
                    try:
                        result = st.session_state.explainer.predict_with_xgboost(skills, top_n=3)
                        results.append({
                            'Skills': skills[:50] + '...' if len(skills) > 50 else skills,
                            'Predicted Role': result['predicted_role'],
                            'Confidence': f"{result['confidence']:.1%}",
                            'Top 2': result['top_predictions'][1]['role'] if len(result['top_predictions']) > 1 else 'N/A',
                            'Top 3': result['top_predictions'][2]['role'] if len(result['top_predictions']) > 2 else 'N/A'
                        })
                    except Exception as e:
                        results.append({
                            'Skills': skills[:50],
                            'Predicted Role': 'Error',
                            'Confidence': '0%',
                            'Top 2': 'N/A',
                            'Top 3': 'N/A'
                        })
                    
                    progress_bar.progress((i + 1) / len(lines))
                
                st.markdown("---")
                st.markdown("### 📊 Batch Results")
                
                import pandas as pd
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No valid skill sets found")
        else:
            st.warning("Please enter at least one skill set")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🤖 <strong>XAI Testing Dashboard</strong> | Built with Streamlit</p>
    <p>Using SHAP & LIME for Explainable AI</p>
</div>
""", unsafe_allow_html=True)
