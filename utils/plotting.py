
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_hist(df: pd.DataFrame, columns: list[str], bins=50, color="skyblue", jitter=False,edgecolor="black"):
    
    """Plot histograms for one or more numeric columns in a dataframe"""

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
    
    """Plot a a vertical bar chart from a dataframe column pair"""
    
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

    plt.title(title if title else f"{y_col} by {x_col}", fontsize=10)
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.xticks(rotation=rotation,fontsize = 10)
    plt.tight_layout()
    plt.show()


def barh_percent(df, col, order=None, palette="Set2", title=None):
    
    """Plot a horizontal bar chart showing the percentage distribution of a categorical column"""
    
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
    
    
    
    
    
def plot_cvd_stacked_bar(df, group_col, 
                         title=None,
                         xlabel=None,
                         colors=("#D15C4F", "#58B153"),
                         figsize=(8,5)):

    # 1. Crosstab normalized to 100%
    ct = pd.crosstab(
        df[group_col],
        df["cardio_label"],
        normalize="index"
    ) * 100

    # 2. Plot
    ax = ct.plot(
        kind="bar",
        stacked=True,
        figsize=figsize,
        color=colors,
        edgecolor="white",
        width=0.75
    )

    # 3. Add percentage labels
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.0f%%",
            label_type="center",
            fontsize=9,
            color="black"
        )

    # 4. Add sample sizes above bars
    totals = df[group_col].value_counts().sort_index()
    for i, total in enumerate(totals):
        ax.text(
            i,
            100,
            f"n={total}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold"
        )

    # 5. Titles and labels
    plt.title(title or f"CVD Prevalence by {group_col} (100%)")
    plt.ylabel("Percent")
    plt.xlabel(xlabel or group_col.replace("_", " ").title())
    plt.legend(title="Cardio")
    plt.tight_layout()
    plt.show()


def plot_box(df, col, target='cardio',palette=None):
    
    """Draws a boxplot of a numeric variable split by the cardio target """
    
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=target, y=col, data=df, palette=palette)
    
    plt.title(f'Distribution of {col} by {target.capitalize()}', fontsize=12)
    plt.xlabel(target.capitalize())
    plt.ylabel(col.capitalize())
    plt.xticks([0, 1], ['No CVD', 'CVD'])
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    

def plot_stacked_counts(
    df,
    vars_,
    target_col,
    *,
    var_order=None,                
    target_order=None,           
    ncols=3,
    colors=("lightcoral", "skyblue"),
    rotation=0,
    show_percent_inside=True,
    show_totals_above=True,
    legend_loc="upper center",
    legend_fontsize=12,
    ymax = None
):
    """Stacked bar subplots with COUNT on y-axis, % labels inside segments, and n above bars """
    
    n = len(vars_)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, var in enumerate(vars_):
        ax = axes[i]

        # --- counts per category x target ---
        ct = pd.crosstab(df[var], df[target_col])

        # enforce row order (per variable) if provided
        
        if var_order and var in var_order:
            ct = ct.reindex(var_order[var])

        # enforce target (column) order if provided
        if target_order is not None:
            present = [c for c in target_order if c in ct.columns]
            ct = ct.reindex(columns=present)

        # plot stacked COUNTS
        ct.plot(kind="bar", stacked=True, ax=ax, legend=False,
                color=list(colors), rot=rotation)
        
        if ymax is not None:
            ax.set_ylim(0, ymax)

        # % labels INSIDE each colored segment
        if show_percent_inside:
            totals = ct.sum(axis=1).values
            for container in ax.containers:
                counts = [patch.get_height() for patch in container]
                perc = [(c/t*100) if t > 0 else 0 for c, t in zip(counts, totals)]
                labels = [f"{p:.0f}%" if c > 0 else "" for p, c in zip(perc, counts)]
                ax.bar_label(container, labels=labels, label_type="center",
                             fontsize=7, color="white")

        # total n ABOVE each bar
        if show_totals_above:
            totals = ct.sum(axis=1).values
            for x, total in enumerate(totals):
                ax.text(x, total * 1.02, f"n={int(total)}",
                        ha="center", va="bottom", fontsize=9)

        # titles/axes
        pretty = var.removesuffix("_label").replace("_", " ").capitalize()
        ax.set_title(pretty)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.grid(False)

    # remove unused axes (if grid not full)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # legend in given order
    cols_for_legend = (list(target_order)
                       if target_order is not None
                       else list(ct.columns))
    fig.legend([str(c) for c in cols_for_legend],
               loc=legend_loc, ncol=len(cols_for_legend),
               frameon=False, fontsize=legend_fontsize)

    plt.tight_layout(rect=[0, 0, 1, 0.93 if "upper" in legend_loc else 1])
    plt.show()



def plot_heatmap_table(df,groupby_columns:list[str],target_col,title,xlabel,ylabel,ax,cmap="Reds"):

    """ Plot a heatmap that shows how the average of a target column changes across two categories"""
    
    heatmap_data = (
        df
        .groupby(groupby_columns, observed=True)[target_col]
        .mean()
        .unstack() * 100
    )

    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap=cmap, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return heatmap_data
    

def plot_categorical_distribution(df, column, palette="Set2"):
    """
    Plots the distribution of a categorical column with percentage labels.
    Works for cholesterol, glucose, smoke, alco, active, etc.
    """
    
    plt.figure(figsize=(6,4))

   
    ax = sns.countplot(
        data=df,
        x=column,
        hue=column,        
        palette=palette,
        legend=False       
    )

    total = len(df)

    # Add % labels
    for bar in ax.patches:
        count = bar.get_height()
        percent = 100 * count / total
        ax.text(
            bar.get_x() + bar.get_width()/2,
            count + (0.01 * total),
            f"{percent:.1f}%",
            ha="center",
            fontsize=10
        )

    plt.title(f"Distribution of {column.replace('_', ' ').title()}")
    plt.xlabel(column.replace('_', ' ').title())
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()



def plot_categorical_subplot(df, columns, palette="Set2"):
    
    """Creates subplots for multiple categorical variables """

    # Compute global max count for consistent scale
    global_max = max(df[col].value_counts().max() for col in columns)

    fig, axes = plt.subplots(1, len(columns), figsize=(6*len(columns), 5))

    for ax, col in zip(axes, columns):

        # countplot with future-safe syntax
        sns.countplot(
            data=df,
            x=col,
            hue=col,
            palette=palette,
            legend=False,
            ax=ax
        )
        
        total = len(df)

        # Add % labels on each bar
        for bar in ax.patches:
            count = bar.get_height()
            percent = 100 * count / total
            ax.text(
                bar.get_x() + bar.get_width()/2,
                count + global_max*0.02,
                f"{percent:.1f}%",
                ha="center",
                fontsize=9
            )

        # Same y-axis for all plots
        ax.set_ylim(0, global_max * 1.15)

        # Titles and labels
        ax.set_title(f"Distribution of {col[:-6].title()}")
        ax.set_xlabel(col.replace('_', ' ').title())
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.show()


