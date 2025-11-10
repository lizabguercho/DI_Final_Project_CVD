
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_hist(df: pd.DataFrame, columns: list[str], bins=50, color="skyblue", jitter=False):
    """
    Plot histograms for one or more numeric columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe.
    columns : list
        List of column names to plot.
    bins : int, optional
        Number of histogram bins (default=50).
    color : str, optional
        Color of the bars (default="skyblue").
    jitter : bool, optional
        If True, adds small random noise to make discrete data look smoother.
    """

    n = len(columns)
    plt.figure(figsize=(6 * n, 4))

    for i, col in enumerate(columns, 1):
        plt.subplot(1, n, i)

        data = df[col].dropna()
        if jitter:
            data = data + np.random.uniform(-0.3, 0.3, len(data))  

        sns.histplot(data, bins=bins, color=color, kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
