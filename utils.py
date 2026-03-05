import matplotlib.pyplot as plt

def plot_missing(df):

    missing = df.isnull().sum()

    plt.figure(figsize=(8,4))
    missing.plot(kind="bar")

    plt.title("Missing Values per Column")
    plt.ylabel("Count")

    return plt
import seaborn as sns

def correlation_heatmap(df):

    numeric_df = df.select_dtypes(include='number')

    if numeric_df.shape[1] < 2:
        return None

    plt.figure(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")

    return plt