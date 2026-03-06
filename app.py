import streamlit as st
import pandas as pd

from analyzer import analyze_dataset
from analyzer import dataset_score
from analyzer import recommendations
from analyzer import dataset_description
from analyzer import detect_outliers
from analyzer import suggest_ml_task
from analyzer import generate_report

from preprocessing_agent import preprocess_dataset
from model_agent import run_model

from utils import plot_missing
from utils import correlation_heatmap

from alerts import send_alert


st.set_page_config(page_title="AI Dataset Quality Agent")

st.title("AI Dataset Quality Agent")

uploaded_file = st.file_uploader("Upload CSV dataset")


if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())


    # -----------------------------
    # DATASET ANALYSIS
    # -----------------------------

    report = analyze_dataset(df)

    st.subheader("Dataset Report")

    st.write("Rows:", report["rows"])
    st.write("Columns:", report["columns"])
    st.write("Missing values:", report["missing_values"])
    st.write("Duplicate rows:", report["duplicate_rows"])
    st.write("Missing value percentage:", round(report["missing_percentage"], 2), "%")

    st.write("Numeric columns:", report["numeric_columns"])
    st.write("Categorical columns:", report["categorical_columns"])


    # -----------------------------
    # VISUALIZATION
    # -----------------------------

    st.subheader("Missing Value Visualization")

    fig = plot_missing(df)
    st.pyplot(fig)


    # -----------------------------
    # QUALITY SCORE
    # -----------------------------

    score = dataset_score(df)

    st.subheader("Dataset Quality Score")
    st.metric("Quality Score", f"{round(score,2)}/100")


    # -----------------------------
    # RECOMMENDATIONS
    # -----------------------------

    tips = recommendations(df)

    st.subheader("AI Recommendations")

    for tip in tips:
        st.write("•", tip)


    # -----------------------------
    # DATASET UNDERSTANDING
    # -----------------------------

    desc = dataset_description(df)

    st.subheader("Dataset Understanding Agent")

    for d in desc:
        st.write("•", d)


    # -----------------------------
    # CORRELATION ANALYSIS
    # -----------------------------

    st.subheader("Feature Relationship Analysis")

    heat = correlation_heatmap(df)

    if heat is not None:
        st.pyplot(heat)
    else:
        st.write("Not enough numeric features for correlation analysis.")


    # -----------------------------
    # OUTLIER DETECTION
    # -----------------------------

    outs = detect_outliers(df)

    st.subheader("Outlier Detection Agent")

    if len(outs) == 0:
        st.write("No major outliers detected.")
    else:

        sorted_outs = sorted(outs.items(), key=lambda x: x[1], reverse=True)
        top_outliers = sorted_outs[:5]

        for col, count in top_outliers:
            st.write(f"{count} potential outliers detected in column: {col}")


    # -----------------------------
    # ML TASK SUGGESTION
    # -----------------------------

    ml_task = suggest_ml_task(df)

    st.subheader("ML Task Suggestion")
    st.write(ml_task)


    st.divider()


    # -----------------------------
    # AUTOMATED ML PIPELINE
    # -----------------------------

    st.subheader("Automated ML Pipeline")

    df_clean = preprocess_dataset(df)

    st.write("Preprocessed dataset saved as **preprocessed_dataset.csv**")


    import os

    file_path = "preprocessed_dataset.csv"

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            st.download_button(
            label="Download Preprocessed Dataset",
            data=f,
            file_name="preprocessed_dataset.csv",
            mime="text/csv"
        )

    else:
        st.warning("Preprocessed dataset not available")

    # run ML model
    try:

        model_name, model_score = run_model(df_clean, ml_task)

        st.write("Model Used:", model_name)
        st.write("Model Score:", model_score)

    except Exception as e:

        st.warning("Automatic ML pipeline could not run.")
        st.write("Reason:", e)


    # -----------------------------
    # GENERATE REPORT + TELEGRAM ALERT
    # -----------------------------

    report_file = generate_report(df, report, score, tips, outs, ml_task)

    st.success("Report generated successfully")

    try:
        send_alert(uploaded_file.name, report_file)
        st.info("Telegram alert sent successfully")

    except Exception as e:
        st.error(f"Telegram alert failed: {e}")