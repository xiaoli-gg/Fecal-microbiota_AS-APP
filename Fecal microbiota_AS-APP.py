import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from io import BytesIO

# 检查必要文件是否存在
@st.cache_resource
def load_model():
    try:
        if not os.path.exists('gbm_model.pkl'):
            st.error("模型文件 'gbm_model.pkl' 不存在，请检查文件路径。")
            st.stop()
        return joblib.load('gbm_model.pkl')
    except Exception as e:
        st.error(f"加载模型时出错: {str(e)}")
        st.stop()

@st.cache_data
def load_background_data():
    try:
        if not os.path.exists('shap_background.csv'):
            st.error("背景数据文件 'shap_background.csv' 不存在，请检查文件路径。")
            st.stop()
        return pd.read_csv("shap_background.csv")
    except Exception as e:
        st.error(f"加载背景数据时出错: {str(e)}")
        st.stop()

# 加载模型和背景数据
model = load_model()
background = load_background_data()

# 特征范围定义（修正了语法错误）
feature_ranges = {
    "Chlamydia.psittaci_83554": {"type": "numerical", "min": 0, "max": 1, "default": 0.00056},
    "Aliarcobacter.cryaerophilus_28198": {"type": "numerical", "min": 0, "max": 1, "default": 0.00002},
    "Collinsella.aerofaciens_74426": {"type": "numerical", "min": 0, "max": 1, "default": 0.00057},  # 修正：添加逗号
    "Bifidobacterium.breve_1685": {"type": "numerical", "min": 0, "max": 1, "default": 0.00006},  # 修正：添加逗号
    "Actinomyces.naeslundii_1655": {"type": "numerical", "min": 0, "max": 1, "default": 0},
    #"Bronchoscopy": {"type": "categorical", "options": [0, 1]},
}

# 设置页面配置
st.set_page_config(
    page_title='AI-assisted ankylosing spondylitis (AS) Prediction Tool',
    page_icon="🩺",
    layout="wide"
)

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
    elif properties["type"] == "categorical":
        value = st.sidebar.selectbox(
            label=f"{feature}",
            options=properties["options"],
        )
    feature_values.append(value)

# 页面底部版权信息
st.sidebar.markdown("---")
st.sidebar.markdown("##### All rights reserved") 
st.sidebar.markdown("##### Contact: mengpanli163@163.com")
st.sidebar.markdown("##### (Mengpan Li, Shanghai Jiao Tong University School of Medicine)")

# 主页面内容
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

# 预测按钮
if st.button("🔍 Predict", type="primary"):
    try:
        # 构造输入
        features = np.array([feature_values])
        feature_df = pd.DataFrame(features, columns=feature_ranges.keys())

        # 预测
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]
        probability = predicted_proba[1] * 100  # 二分类中通常 [1] 为"阳性"类

        # 显示预测结果
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        # 使用列布局显示结果
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if probability > 50:
                st.error(f"⚠️ High Risk: {probability:.2f}% probability of AS")
            else:
                st.success(f"✅ Low Risk: {probability:.2f}% probability of AS")

        # 创建概率条形图
        st.markdown("### 📈 Probability Distribution")
        
        # 模拟预测概率
        sample_prob = {
            'Not AS': predicted_proba[0],  # 未患病概率
            'AS': predicted_proba[1]   # 患病概率
        }

        # 使用 Streamlit 原生图表
        prob_df = pd.DataFrame({
            'Class': ['Not AS', 'AS'],
            'Probability': [sample_prob['Not AS'], sample_prob['AS']]
        })
        
        st.bar_chart(prob_df.set_index('Class'))

        # 创建更详细的 matplotlib 图表
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.barh(['Not Ankylosing Spondylitis', 'Ankylosing Spondylitis'], 
                       [sample_prob['Not AS'], sample_prob['AS']], 
                       color=['#4CAF50', '#F44336'])

        ax.set_title("Prediction Probability", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Probability", fontsize=12, fontweight='bold')
        ax.set_ylabel("Classes", fontsize=12, fontweight='bold')
        
        # 添加数值标签
        for i, v in enumerate([sample_prob['Not AS'], sample_prob['AS']]):
            ax.text(v + 0.02, i, f"{v:.3f}", va='center', fontsize=12, fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # SHAP 分析
        st.markdown("### 🔍 Feature Importance Analysis (SHAP)")
        
        with st.spinner("Calculating SHAP values..."):
            try:
                # 根据模型类型选择合适的 explainer
                # 如果是树模型，使用 TreeExplainer；否则使用 KernelExplainer
                model_type = str(type(model)).lower()
                
                if 'tree' in model_type or 'forest' in model_type or 'gradient' in model_type or 'xgb' in model_type or 'lgb' in model_type:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(feature_df)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # 取正类的 SHAP 值
                else:
                    explainer = shap.KernelExplainer(model.predict_proba, background.sample(100))
                    shap_values = explainer.shap_values(feature_df)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # 取正类的 SHAP 值

                # 创建 SHAP 图
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # 使用 waterfall plot 替代 force plot（更适合单个样本）
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
                        data=feature_df.iloc[0],
                        feature_names=list(feature_ranges.keys())
                    ),
                    show=False
                )
                
                st.pyplot(plt.gcf())
                plt.close()
                
            except Exception as e:
                st.warning(f"SHAP analysis failed: {str(e)}")
                st.info("SHAP analysis requires compatible model types and may not work with all models.")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.info("Please check your input values and try again.")

# 添加使用说明
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

# 添加免责声明
st.markdown("---")
st.markdown("**Disclaimer**: This AI tool is designed for research and educational purposes only. "
           "It should not be used as a substitute for professional medical advice, diagnosis, or treatment.")