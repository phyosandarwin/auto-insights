import streamlit as st
from pycaret.regression import *

def train_reg_model(df, target_var, metric):
    st.subheader("Preprocessing data:")
    
    setup(df, target=target_var, normalize=True, normalize_method='robust', 
          numeric_imputation='median', session_id=100, fold=3)
    setup_df = pull()
    st.dataframe(setup_df, use_container_width=True)
    
    st.subheader("Performance Metrics")
    with st.spinner(text = "Evaluating regression models..."):
        # Compare models and display performance metrics
        best_reg_model = compare_models(sort=metric)
    st.toast("Evaluated all models. Check out the performance!", icon="😍")
    compare_df = pull()
    st.dataframe(compare_df, use_container_width=True)
    st.write(best_reg_model)

    st.subheader("Hyperparameter tuning")
    with st.spinner("Tuning hyperparameters, wait for it..."):
        tuned_reg_model = tune_model(best_reg_model, optimize=metric)
    
    st.toast("Tuned all the hyperparameters!", icon="🥳")
    tune_trials_df = pull()
    st.dataframe(tune_trials_df, use_container_width=True)

    st.subheader("Visualise Performance of Tuned model")
    plot_model(tuned_reg_model, plot ='residuals', display_format='streamlit', scale=0.75)
    plot_model(tuned_reg_model, plot = 'error', display_format='streamlit', scale=0.75)
    plot_model(tuned_reg_model, plot="feature", display_format='streamlit', scale=0.75)

    st.subheader("Predictions on holdout set", divider='gray')
    predictions = predict_model(tuned_reg_model)
    st.dataframe(predictions, use_container_width=True)

    # Return the tuned model
    return tuned_reg_model