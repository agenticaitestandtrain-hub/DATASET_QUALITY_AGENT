import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def preprocess_dataset(df):

    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    if len(cat_cols) > 0:

        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

        for c in cat_cols:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c])

    df.to_csv("preprocessed_dataset.csv", index=False)

    return df