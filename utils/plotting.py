
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_hist(df: pd.DataFrame, columns: list[str], bins=50, color="skyblue", jitter=False,edgecolor="black"):
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

        sns.histplot(data, bins=bins, color=color,edgecolor=edgecolor, kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")

    plt.tight_layout()
    plt.show()




def plot_bar(df, x_col, y_col, title=None, xlabel=None, ylabel=None, rotation=0, color=None):
    """
    Plot a bar chart for any DataFrame columns.

    Parameters:
        df (pd.DataFrame): Your DataFrame
        x_col (str): Column name for the x-axis
        y_col (str): Column name for the y-axis
        title (str, optional): Title of the chart
        xlabel (str, optional): Label for x-axis
        ylabel (str, optional): Label for y-axis
        rotation (int, optional): Rotation of x-axis labels
        color (str, optional): Bar color (default Matplotlib color cycle)
    """
    plt.rcParams["axes.grid"] = False 
    plt.figure(figsize=(8, 5))
    plt.bar(df[x_col], df[y_col], color=color)
    plt.title(title if title else f"{y_col} by {x_col}", fontsize=14)
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()

def plot_bar2(df, x_col, y_col, title=None, xlabel=None, ylabel=None,
             rotation=0, color=None, palette="Set2", show_values=True):
    plt.rcParams["axes.grid"] = False 
    plt.figure(figsize=(8, 5))

    if color is None:
        colors = sns.color_palette(palette, n_colors=len(df))
    else:
        colors = color

    bars = plt.bar(df[x_col], df[y_col], color=colors)

    if show_values:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height,
                     f"{int(height)}", ha='center', va='bottom', fontsize=9)

    plt.title(title if title else f"{y_col} by {x_col}", fontsize=14)
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


def plot_pie(df, col, title=None, colors=None, palette="Set3",distance=1.12, textprops=None):
    """
    Plot a pie chart showing percentage and count for each category.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        col (str): Column name to plot.
        title (str, optional): Chart title.
        colors (list, optional): Custom color list.
        palette (str, optional): Seaborn palette name (default: 'Set3').
    """
    counts = df[col].value_counts(sort=False)
    total = counts.sum()

    # Auto-generate colors if none provided
    if colors is None:
        colors = sns.color_palette(palette, n_colors=len(counts))

    # Custom autopct function to show both % and counts
    def autopct_format(pct):
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n(n={count})"

    plt.figure(figsize=(6, 6))
    plt.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct=autopct_format,   # 👈 custom function
        startangle=90,
        counterclock=False
    )
    plt.title(title if title else f"Distribution of {col}", fontsize=14)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_pie2(df, col, title=None, colors=None, palette="Set3", distance=1.2,pctdistance=0.8, textprops=None):
    """
    Plot a pie chart with percentage and count labels placed outside the pie.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        col (str): Column name to plot.
        title (str, optional): Chart title.
        colors (list, optional): Custom color list.
        palette (str, optional): Seaborn palette name (default: 'Set3').
        distance (float): Distance of label text from pie center (default 1.12).
        textprops (dict): Text style properties for labels.
    """
    counts = df[col].value_counts(sort=False)
    total = counts.sum()

    # Auto-generate colors if none provided
    if colors is None:
        colors = sns.color_palette(palette, n_colors=len(counts))

    # Default text style
    if textprops is None:
        textprops = {"fontsize": 10, "weight": "bold"}

    # Label text builder
    labels = [
        f"{cat}\n{(count / total) * 100:.1f}% (n={count})"
        for cat, count in zip(counts.index, counts.values)
    ]

    # Plot
    plt.figure(figsize=(7, 7))
    wedges, texts = plt.pie(
        counts.values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.9, edgecolor="white")
    )

    # Place labels outside with lines
    for i, (wedge, label) in enumerate(zip(wedges, labels)):
        ang = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.deg2rad(ang)) * distance
        y = np.sin(np.deg2rad(ang)) * distance
        ha = "left" if x > 0 else "right"
        plt.text(x, y, label, ha=ha, va="center", **textprops)
        plt.plot([x * 0.92, x], [y * 0.92, y], color="gray", lw=0.8)

    plt.title(title if title else f"Distribution of {col}", fontsize=14)
    plt.tight_layout()
    plt.show()

def barh_percent(df, col, order=None, palette="Set2", title=None):
    counts = df[col].value_counts()
    if order is not None:
        counts = counts.reindex(order).dropna()
    perc = (counts / counts.sum() * 100).round(1)

    plt.figure(figsize=(8, 4.5))
    colors = sns.color_palette(palette, n_colors=len(counts))
    bars = plt.barh(counts.index, perc, color=colors)
    for b, p in zip(bars, perc):
        plt.text(b.get_width()+0.5, b.get_y()+b.get_height()/2, f"{p}%", va="center")
    plt.title(title or f"{col} distribution (%)")
    plt.xlabel("Percent")
    plt.xlim(0, max(perc)*1.15)
    plt.tight_layout()
    plt.show()