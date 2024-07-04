import os
import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
from helpers.reg_helpers import *
from helpers.classifn_helpers import *


st.set_page_config(page_title="AutoInsights", layout='wide', page_icon="⚙️", initial_sidebar_state='expanded')

st.logo("assets/logo.png")
def clear_files():
    pickle_files = [file for file in os.listdir('.') if file.endswith('.pkl')]
    for file_name in pickle_files:
        os.remove(file_name)
    if not any(file.endswith('.pkl') for file in os.listdir('.')):
        st.toast(":green[Removed all models!]", icon="✅")
    else:
        st.toast(":red[Error removing model files]", icon="🚨")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 350px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)
### STREAMLIT UI/ VIEW ###
with st.sidebar:
    st.title("AutoInsights")
    st.info("An AutoML platform to faciliate your classification and regression tasks.")
    with st.container(border=True):
        st.subheader("Get started here!")
        task_choice = st.radio("Choose your task:", options=["Regression", "Classification"], horizontal=True)
        restart_button = st.button("Restart Tasks 🔄", use_container_width=True, type="primary")
        if restart_button:
            clear_files()
    
    with st.expander("More:", icon="👇"):
        st.markdown("**Check out project on [Github](https://github.com/phyosandarwin/auto-insights)** 🤓")
        st.markdown(
                """
                **Connect with me via:**

                [![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=flat&logo=github)](https://github.com/phyosandarwin)
                [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://linkedin.com/in/phyosandarwin)
                """
    )


st.markdown(f'''<span style="font-size: 20px;">Getting started on your :blue-background[**{task_choice}**] Task!</span>''', unsafe_allow_html=True)

# Task specific sections
if task_choice == "Regression":
    with st.container(border=True):
        st.subheader("Step 1: Upload dataset 📤", divider='blue')
        file = st.file_uploader(label="Upload regression dataset:", type="csv")
        if file:
            reg_df = pd.read_csv(file, index_col=None)
            st.dataframe(reg_df.head(), use_container_width=True)
            if not reg_df.empty:
                st.subheader("Step 2: View Data Profile 🔎", divider='blue')
                profile_df = ProfileReport(reg_df, minimal=True, explorative=True)
                st_profile_report(profile_df)
                
                st.subheader("Step 3: Model Training and Evaluation 🚀", divider='blue')
                metric = st.selectbox("Choose evaluation metric for regression:", ["R2", "MAE", "MSE", "RMSE", "MAPE"])
                target = st.selectbox("Choose target column", reg_df.columns)
                    
                if st.button("Run Modelling"):
                    best_model = train_reg_model(reg_df, reg_df[target], metric)
                    save_model(best_model, 'best_reg_model')

                    if os.path.exists('best_reg_model.pkl'):
                        with open('best_reg_model.pkl', 'rb') as f:
                            st.download_button('📥Download Model', f, file_name='best_reg_model.pkl')

elif task_choice == "Classification":
    with st.container(border=True):
        st.subheader("Step 1: Upload dataset 📤", divider='blue')
        file = st.file_uploader(label="Upload tabular classification dataset:", type="csv")
        if file:
            class_df = pd.read_csv(file, index_col=None)
            st.dataframe(class_df.head(), use_container_width=True)
            if not class_df.empty:
                st.subheader("Step 2: View Data Profile 🔎", divider='blue')
                profile_df = ProfileReport(class_df, minimal=True, explorative=True)
                st_profile_report(profile_df)
                
                st.subheader("Step 3: Model Training and Evaluation 🚀", divider='blue')
                metric = st.selectbox("Choose evaluation metric for classification:", ["Accuracy", "AUC", "F1"])
                target = st.selectbox("Choose the target column", class_df.columns)
                    
                if st.button("Run Modelling"):
                    best_model = train_class_model(class_df, class_df[target], metric)
                    save_model(best_model, 'best_class_model')

                    if os.path.exists('best_class_model.pkl'):
                        with open('best_class_model.pkl', 'rb') as f:
                            st.download_button('📥Download Model', f, file_name='best_class_model.pkl')
    
    
