import streamlit as st
from pycaret.classification import *


def train_class_model(df, target_var, metric):
    # Set up PyCaret environment
    st.subheader("Preprocessing data:")
    setup(df, target=target_var, normalize=True, normalize_method='robust', numeric_imputation='median',
          data_split_stratify=True, fix_imbalance=True, feature_selection=True)
    
    setup_df = pull()
    st.dataframe(setup_df, use_container_width=True)

    st.subheader("Performance Metrics")
    
    with st.spinner(text = "Evaluating classifiers..."):
        # Compare models and display performance metrics
        best_class_model = compare_models(sort=metric)
    st.toast("Evaluated all models. Check out the performance!", icon="😍")
    compare_df = pull()
    st.dataframe(compare_df, use_container_width=True)
    st.write(best_class_model)

    st.subheader("Hyperparameter tuning")
    with st.spinner("Tuning hyperparameters, wait for it..."):
        tuned_class_model = tune_model(best_class_model, optimize=metric, 
                                       search_library='scikit-optimize', search_algorithm='bayesian')
    
    st.toast("Tuned all the hyperparameters!", icon="🥳")
    tune_trials_df = pull()
    st.dataframe(tune_trials_df, use_container_width=True)


    st.subheader("Visualise Performance of Tuned model")
    plot_model(tuned_class_model, plot="class_report", display_format='streamlit', scale=0.75)
    plot_model(tuned_class_model, plot="confusion_matrix", display_format='streamlit', scale=0.75)
    if hasattr(tuned_class_model, "predict_proba"):
        plot_model(tuned_class_model, plot="auc", display_format='streamlit', scale=0.75)

    st.subheader("Predictions on holdout set", divider='gray')
    predictions = predict_model(tuned_class_model)
    st.dataframe(predictions, use_container_width=True)

    # Return the tuned model
    return tuned_class_model
