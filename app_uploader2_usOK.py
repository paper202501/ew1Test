# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from app_fx1 import process_features

# 定义默认路径
# DEFAULT_IMG = "/home/cc/ff_item/py310ML/_webApp_streamlit_Ants/sub1ImgNew.BMP"
# DEFAULT_ROI = "/home/cc/ff_item/py310ML/_webApp_streamlit_Ants/sub1RoiNew.nii.gz"
# p_lr_model = "/home/cc/ff_item/py310ML/_webApp_streamlit_Ants/save_LR.joblib"
# p_bg = "/home/cc/ff_item/py310ML/_webApp_streamlit_Ants/save_bg.joblib"
DEFAULT_IMG = "sub1ImgNew.BMP"
DEFAULT_ROI = "sub1RoiNew.nii.gz"
p_lr_model = "save_LR.joblib"
p_bg = "save_bg.joblib"


def save_as_niigz(uploaded_file):
    """保存为临时 .nii.gz 文件"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


# --- UI 布局 ---
st.title("RAIR USImg Predictor")
age_val = st.number_input("Age:", value=57)  # 年龄输入
c1, c2 = st.columns(2)
with c1:
    up_img = st.file_uploader("Upload Image (Optional)", type=["bmp", "png", "dicom"])
    # 逻辑判断：优先使用上传的文件，否则使用默认路径
    if up_img is not None:
        p_img = save_as_niigz(up_img)  # 使用之前定义的保存函数
        st.info(f"Using uploaded image: {up_img.name}")
    else:
        p_img = DEFAULT_IMG
        st.caption(f"Using default: {os.path.basename(DEFAULT_IMG)}")
with c2:
    up_roi = st.file_uploader("Upload Mask (Optional)", type=[".nii.gz"])
    if up_roi is not None:
        p_roi = save_as_niigz(up_roi)
        st.info(f"Using uploaded mask: {up_roi.name}")
    else:
        p_roi = DEFAULT_ROI
        st.caption(f"Using default: {os.path.basename(DEFAULT_ROI)}")

if st.button("featureEx & Predict"):
    try:
        # 执行特征工程流水线
        x_input = process_features(p_img, p_roi, age_val)
        lr_model = joblib.load(p_lr_model)

        # 模型推理 (使用 .values 确保纯数据输入)
        y_cls = lr_model.predict(x_input)[0]
        y_prob = lr_model.predict_proba(x_input)[0]

        # 结果展示
        st.divider()
        res_label = "Positive (+)  >>>  RAIR" if y_cls == 1 else "Negative (-)  >>>  noRAIR"
        # st.subheader(f"Prediction: {res_label}")
        # st.metric("Confidence", f"{y_prob[y_cls] * 100:.2f}%")
        st.subheader(f"Prediction: {res_label}   |   Confidence: {y_prob[y_cls] * 100:.2f}%")

        # SHAP 水瀑图可视化
        st.subheader("SHAP Waterfall Plot for New Sample")

        # 使用 SHAP 解释预测
        x_shap = np.array(x_input)
        f_cols = [
            'Age', 'ResNetPCA51', 'ResNetPCA2', 'ViTPCA67', 'ViTPCA44', 'ResNetPCA73',
            'ResNetPCA52', 'ViTPCA1', 'ResNetPCA13', 'log-sigma-1-mm-2D_ngtdm_Strength',
            'log-sigma-4-mm-2D_glrlm_ShortRunLowGrayLevelEmphasis', 'ViTPCA25',
            'log-sigma-1-mm-2D_glcm_Imc2', 'ViTPCA32', 'ResNetPCA48'
        ]

        X_train_scaled = joblib.load(p_bg)
        explainer = shap.LinearExplainer(lr_model, X_train_scaled)
        x_shap = pd.DataFrame(x_shap, columns=f_cols)
        shap_values = explainer(x_shap, )
        shap.plots.waterfall(shap_values[0], max_display=12, show=True, )

        plt.savefig("shap.png", bbox_inches='tight', format='png')
        st.image("shap.png")

    except Exception as e:
        st.error(f"Processing Error: {str(e)}")

"""
    2026-2-14 05:03:28
        conda activate py310brain
        streamlit run /home/cc/ff_item/py310ML/_webApp_streamlit_Ants/app_uploader2_usOK.py
"""
