import pandas as pd


def analyze_dataset(df):

    report = {}

    report["rows"] = df.shape[0]
    report["columns"] = df.shape[1]

    missing = df.isnull().sum()

    report["missing_values"] = missing.sum()

    report["missing_percentage"] = (missing.sum() / (df.shape[0] * df.shape[1])) * 100

    report["duplicate_rows"] = df.duplicated().sum()

    report["numeric_columns"] = list(df.select_dtypes(include='number').columns)

    report["categorical_columns"] = list(df.select_dtypes(include='object').columns)

    return report
def dataset_score(df):

    score = 100

    rows, cols = df.shape

    # Missing values penalty
    missing_pct = df.isnull().sum().sum() / (rows * cols)
    score -= missing_pct * 40

    # Duplicate rows penalty
    dup_pct = df.duplicated().sum() / rows
    score -= dup_pct * 30

    # Outlier penalty (numeric columns)
    numeric_cols = df.select_dtypes(include=["int64","float64"])

    outlier_count = 0

    for col in numeric_cols.columns:

        Q1 = numeric_cols[col].quantile(0.25)
        Q3 = numeric_cols[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outlier_count += ((numeric_cols[col] < lower) | (numeric_cols[col] > upper)).sum()

    outlier_ratio = outlier_count / (rows * len(numeric_cols.columns) + 1)

    score -= outlier_ratio * 30

    return max(round(score,2),0)
def recommendations(df):

    tips = []

    missing = df.isnull().sum()
    total_missing = missing.sum()

    if total_missing > 0:
        cols = list(missing[missing > 0].index)
        tips.append(f"Missing values detected in columns: {cols}. Consider imputation or removing affected rows.")

    duplicates = df.duplicated().sum()

    if duplicates > 0:
        tips.append(f"{duplicates} duplicate rows detected. Consider dropping duplicates.")

    numeric_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include='object').columns

    if len(cat_cols) > 0:
        tips.append(f"Categorical columns found: {list(cat_cols)}. Encoding may be required before ML.")

    if len(numeric_cols) > 0:
        tips.append("Numeric features detected. Consider scaling or normalization for certain ML models.")

    if df.shape[0] < 100:
        tips.append("Dataset is small. Consider collecting more data for better model performance.")

    return tips
def dataset_summary(df):

    summary = {}

    summary["rows"] = df.shape[0]
    summary["columns"] = df.shape[1]

    summary["missing_values"] = df.isnull().sum().sum()
    summary["duplicate_rows"] = df.duplicated().sum()

    summary["numeric_columns"] = list(df.select_dtypes(include='number').columns)
    summary["categorical_columns"] = list(df.select_dtypes(include='object').columns)

    return summary
def dataset_description(df):

    desc = []

    rows, cols = df.shape
    desc.append(f"The dataset contains {rows} rows and {cols} columns.")

    num_cols = list(df.select_dtypes(include='number').columns)
    cat_cols = list(df.select_dtypes(include='object').columns)

    desc.append(f"There are {len(num_cols)} numeric features and {len(cat_cols)} categorical features.")

    if len(num_cols) > 0:
        desc.append("Numeric columns may be useful for regression or clustering models.")

    if len(cat_cols) > 0:
        desc.append("Categorical columns may require encoding before machine learning.")

    return desc
def detect_outliers(df):

    outliers = {}

    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        count = ((df[col] < lower) | (df[col] > upper)).sum()

        if count > 0:
            outliers[col] = count

    return outliers
def suggest_ml_task(df):

    # check for time column
    time_cols = ["date", "time", "timestamp"]

    for col in time_cols:
        if col in df.columns:
            return "Temporal feature detected. Time-series forecasting models may be appropriate."

    # detect potential target column
    last_col = df.columns[-1]

    unique_vals = df[last_col].nunique()

    # classification if few unique values
    if unique_vals < 15 and df[last_col].dtype == "object":
        return "Target column appears categorical. Classification models may be suitable."

    # regression if numeric with many values
    if df[last_col].dtype != "object":
        return "Target column appears continuous. Regression models may be suitable."

    return "Unsupervised learning such as clustering may also be explored."
def generate_report(df, report, score, tips, outs, ml_task):

    text = []

    text.append("DATASET INTELLIGENCE REPORT\n")
    text.append("================================\n")

    text.append(f"Rows: {report['rows']}")
    text.append(f"Columns: {report['columns']}")
    text.append(f"Missing Values: {report['missing_values']}")
    text.append(f"Missing Percentage: {round(report['missing_percentage'],2)}%")
    text.append(f"Duplicate Rows: {report['duplicate_rows']}\n")

    text.append(f"Dataset Quality Score: {round(score,2)}/100\n")

    text.append("RECOMMENDATIONS\n")
    for tip in tips:
        text.append(f"- {tip}")

    text.append("\nOUTLIERS DETECTED\n")

    sorted_outs = sorted(outs.items(), key=lambda x: x[1], reverse=True)[:5]

    for col, count in sorted_outs:
        text.append(f"{col}: {count} outliers")

    text.append("\nML TASK SUGGESTION\n")
    text.append(ml_task)

    report_text = "\n".join(text)

    with open("dataset_report.txt", "w") as f:
        f.write(report_text)

    return "dataset_report.txt"