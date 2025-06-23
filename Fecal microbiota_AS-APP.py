import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

# ---------------- 页面设置 ---------------- #
st.set_page_config(
    page_title='AI-assisted ankylosing spondylitis (AS) Prediction Tool',
    page_icon="🩺",
    layout="wide"
)

# ---------------- 模型和背景数据加载 ---------------- #
@st.cache_resource
def load_model():
    if not os.path.exists('gbm_model.pkl'):
        st.error("模型文件 'gbm_model.pkl' 不存在，请检查文件路径。")
        st.stop()
    return joblib.load('gbm_model.pkl')

@st.cache_data
def load_background_data():
    if not os.path.exists('shap_background.csv'):
        st.error("背景数据文件 'shap_background.csv' 不存在，请检查文件路径。")
        st.stop()
    return pd.read_csv("shap_background.csv")

model = load_model()
background = load_background_data()

# ---------------- 特征定义 ---------------- #
feature_ranges = {
    "Chlamydia.psittaci_83554": {"type": "numerical", "min": 0, "max": 1, "default": 0.00056},
    "Anaerotignum.sp..MB30.C6_3070814": {"type": "numerical", "min": 0, "max": 1, "default": 0.00002},
    "Aliarcobacter.cryaerophilus_28198": {"type": "numerical", "min": 0, "max": 1, "default": 0.00057},
    "Bifidobacterium.breve_1685": {"type": "numerical", "min": 0, "max": 1, "default": 0.00006},
    "Phascolarctobacterium.faecium_33025": {"type": "numerical", "min": 0, "max": 1, "default": 0},
}

# ---------------- 页面布局 ---------------- #
st.title("AI-Assisted Prediction of Ankylosing Spondylitis (AS) Based on Fecal Microbiota")

# 左侧输入面板
st.sidebar.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.sidebar.number_input(
            label=f"{feature}",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
            format="%.6f",
            help=f"Range: {properties['min']} - {properties['max']}"
        )
    feature_values.append(value)

st.sidebar.markdown("---")
st.sidebar.markdown("##### All rights reserved") 
st.sidebar.markdown("##### Contact: mengpanli163@163.com")
st.sidebar.markdown("##### (Mengpan Li, Shanghai Jiao Tong University School of Medicine)")

# 主内容
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### 📊 Model Information")
    st.info("""
        This AI tool predicts the risk of Ankylosing Spondylitis (AS) based on fecal microbiota analysis.
        Please enter the microbiota abundance values in the sidebar and click 'Predict' to get results.
    """)
with col2:
    st.markdown("### 🔬 Features")
    for feature in feature_ranges.keys():
        st.text(f"• {feature}")

# ---------------- 预测逻辑 ---------------- #
if st.button("🔍 Predict", type="primary"):
    try:
        # 构建输入数据
        features = np.array([feature_values])
        feature_df = pd.DataFrame(features, columns=feature_ranges.keys())

        # 预测输出
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]
        probability = predicted_proba[1] * 100  # 预测为 AS 的概率

        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if probability > 50:
                st.error(f"⚠️ High Risk: {probability:.2f}% probability of AS")
            else:
                st.success(f"✅ Low Risk: {probability:.2f}% probability of AS")

        # 概率条形图（Streamlit 和 Matplotlib 双版本）
        st.markdown("### 📈 Probability Distribution")
        prob_df = pd.DataFrame({
            'Class': ['Not AS', 'AS'],
            'Probability': [predicted_proba[0], predicted_proba[1]]
        })
        st.bar_chart(prob_df.set_index('Class'))

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.barh(['Not Ankylosing Spondylitis', 'Ankylosing Spondylitis'],
                       [predicted_proba[0], predicted_proba[1]],
                       color=['#4CAF50', '#F44336'])
        ax.set_title("Prediction Probability", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Probability", fontsize=12)
        ax.set_xlim(0, 1)

        for i, v in enumerate([predicted_proba[0], predicted_proba[1]]):
            ax.text(v + 0.02, i, f"{v:.3f}", va='center', fontsize=12, fontweight='bold')

        st.pyplot(fig)
        plt.close()

        # ---------------- SHAP 分析 ---------------- #
        st.markdown("### 🔍 Feature Importance Analysis (SHAP)")
        with st.spinner("Calculating SHAP values..."):
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(feature_df)
            shap_fig = shap.plots.force(
                explainer.expected_value, shap_values[0], feature_df.iloc[0], matplotlib=True, show=False
            )
            plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
            st.image("shap_force_plot.png")
    
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.info("Please check your input values and try again.")

# ---------------- 使用说明与免责声明 ---------------- #
with st.expander("ℹ️ How to use this tool"):
    st.markdown("""
    1. **Enter feature values**: Use the sidebar to input microbiota abundance values  
    2. **Click Predict**: Press the prediction button to get results  
    3. **Interpret results**:  
       - Green indicates low risk (< 50% probability)  
       - Red indicates high risk (≥ 50% probability)  
    4. **Review SHAP analysis**: Understand which features contribute most to the prediction  

    **Note**: This tool is for research purposes only and should not replace professional medical diagnosis.
    """)

st.markdown("---")
st.markdown("**Disclaimer**: This AI tool is designed for research and educational purposes only. "
            "It should not be used as a substitute for professional medical advice, diagnosis, or treatment.")
