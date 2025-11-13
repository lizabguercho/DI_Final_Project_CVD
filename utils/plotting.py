
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




def plot_bar(df, x_col, y_col, title=None, xlabel=None, ylabel=None,
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
    

def plot_box(df, col, target='cardio',palette=None):
    """
    Draws a boxplot of a numeric variable split by the cardio target.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing your data.
    col : str
        The name of the numeric column to plot.
    target : str, default='cardio'
        The binary target variable (0 = No CVD, 1 = CVD).
    """
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=target, y=col, data=df, palette=palette)
    
    plt.title(f'Distribution of {col} by {target.capitalize()}', fontsize=12)
    plt.xlabel(target.capitalize())
    plt.ylabel(col.capitalize())
    plt.xticks([0, 1], ['No CVD', 'CVD'])
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    

def plot_boxes(df, cols, target='cardio', palette=None, n_cols=3):
    """
    Draws boxplots for multiple numeric columns split by the target variable.
    Compatible with Seaborn 0.13+ (no palette/hue warning).
    """
    n_rows = -(-len(cols) // n_cols)  # ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        # use hue=target and legend=False (recommended by new seaborn)
        sns.boxplot(
            data=df,
            x=target,
            y=col,
            hue=target,
            palette=palette,
            ax=axes[i],
            showfliers=False,
            legend=False
        )

        axes[i].set_title(f'{col.capitalize()} by {target.capitalize()}')
        axes[i].set_xlabel(target.capitalize())
        axes[i].set_ylabel(col.capitalize())

        # ensure fixed tick positions & labels
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['No CVD', 'CVD'])
        axes[i].grid(axis='y', linestyle='--', alpha=0.6)

    # remove unused subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()