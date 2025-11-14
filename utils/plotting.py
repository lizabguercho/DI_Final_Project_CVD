
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
    
    
    
def plot_subplots(df, vars_, target_col=None,*,var_order=None,target_order =(0,1), ncols=3, percent =True,colors=None):
    n = len(vars_)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
    axes = axes.flatten()
    
    for i, var in enumerate(vars_):
        ax = axes[i]
        
        
        if target_col is None:
        # Just count values of the variable itself
            ct = df[var].value_counts(normalize=percent) * (100 if percent else 1)
            ct = pd.DataFrame({var: ct}).T
            colors = ["#4C72B0"]
        
        ct = pd.crosstab(df[var], df[target_col])
        present = [c for c in target_order if c in ct.columns]
        ct = ct.reindex(columns=present)
        if percent:
            ct = ct.div(ct.sum(axis=1), axis=0).fillna(0) * 100
        colors = colors
        
        
        if var_order and var in var_order:
            ct = ct.reindex(var_order[var])
            
        ct.plot(kind="bar", stacked=True, ax=ax, legend=False, color=colors,rot =0)
        
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f%%', label_type='center',
                        fontsize=11, color='white', weight='bold')
        ax.set_title(f"{var[:-6].capitalize()}")
        ax.set_ylabel("Percentage"if percent else "Count")
        ax.set_xlabel("")
        ax.tick_params(axis='x', labelrotation=0)
        ax.grid(False)

    # Remove empty axes if the grid is not full
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add one shared legend on top
    if target_col is not None:
        legend_labels = [str(c) for c in target_order if c in df[target_col].unique()]
        fig.legend(
            legend_labels,
            loc="upper center",
            ncol=len(legend_labels),
            frameon=False,
            fontsize=12
        )
    
    
 

def plot_stacked_counts(
    df,
    vars_,
    target_col,
    *,
    var_order=None,                 # {"cholesterol_label":[...], "glucose_label":[...]}
    target_order=None,              # e.g. (0,1) or ("Female","Male"); if None -> data order
    ncols=3,
    colors=("lightcoral", "skyblue"),
    rotation=0,
    show_percent_inside=True,
    show_totals_above=True,
    legend_loc="upper center",
    legend_fontsize=12,
    ymax = None
):
    """
    Stacked bar subplots with COUNT on y-axis, % labels inside segments, and n above bars.

    Parameters
    ----------
    df : DataFrame
    vars_ : list[str]
        Categorical columns to plot on x-axis (one subplot per variable).
    target_col : str
        Column to stack by (e.g., 'cardio', 'gender_label', 'smoke').
    var_order : dict[str, list[str]], optional
        Desired category order per variable (row order).
    target_order : sequence, optional
        Desired order of stacked categories (column order).
    """
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

    
