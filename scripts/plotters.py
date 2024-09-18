import pandas as pd
import seaborn as sns
from typing import Literal
from pathlib import Path
import matplotlib.pyplot as plt

from scripts import logger
from scripts.utils import get_bins


def custom_hist_multiplot(
    data: pd.DataFrame,
    columns: list[str],
    hue: str | None = None,
    hue_order: list | None = None,
    title: str | None = None,
    features_kind: Literal["num", "cat"] = "num",
    cat_orient: Literal["v", "h"] = "v",
    savepath: Path | None = None,
) -> None:
    """
    Generates a multi-panel histogram plot for the given data and
    columns. Optionally saves the figure to a file.

    Parameters:
        data (pd.DataFrame):
            The input data.
        columns (list[str]):
            The columns to plot histograms for.
        hue (str | None, optional):
            The column to use for grouping the data. Defaults to None.
        hue_order (list | None, optional):
            The order of the hue values to display. Defaults to None.
        title (str | None, optional):
            The title of the plot. Defaults to None.
        features_kind (Literal["num", "cat"], optional):
            Whether the columns are numeric or categorical. Defaults
            to "num".
        cat_orient (Literal["v", "h"], optional):
            The orientation of the categorical histograms, vertical or
            horizontal. Defaults to "v".
        savepath (Path | None, optional):
            The path to save the plot to. Defaults to None.
    """
    if features_kind == "num":
        fig, axs = plt.subplots(len(columns), figsize=(15, 4 * len(columns)))
        if len(columns) == 1:
            axs = [axs]
        if title:
            axs[0].set_title(title)
        for i, col in enumerate(columns):
            sns.histplot(
                data=data,
                x=col,
                ax=axs[i],
                hue=hue,
                hue_order=hue_order,
                common_norm=False,
                fill=True,
                kde=True,
                alpha=0.6,
                stat="density",
                bins=get_bins(len(data)),
            )
            sns.move_legend(axs[i], "upper left", bbox_to_anchor=(1, 1))
    elif features_kind == "cat":
        if cat_orient == "h":
            nunique = sum([data[col].nunique() for col in columns])
            ratios = [data[col].nunique() / nunique for col in columns]
            fig, axs = plt.subplots(
                len(columns),
                figsize=(10, 0.55 * nunique),
                gridspec_kw={"height_ratios": ratios},
            )
        elif cat_orient == "v":
            fig, axs = plt.subplots(
                len(columns), figsize=(15, 4 * len(columns))
            )
        if len(columns) == 1:
            axs = [axs]
        if title:
            fig.suptitle(title, fontsize=16)
        for it, col in enumerate(columns):
            x = col if cat_orient == "v" else None
            y = col if cat_orient == "h" else None
            sns.histplot(
                data=data,
                x=x,
                y=y,
                discrete=True,
                hue=hue,
                hue_order=hue_order,
                common_norm=False,
                multiple="dodge",
                stat="density",
                shrink=0.8,
                legend="full",
                ax=axs[it],
            )
            if cat_orient == "h":
                axs[it].set_ylabel("")
                axs[it].set_title(y)
                if it < len(columns) - 1:
                    axs[it].set_xlabel("")
                axs[it].set_yticks(sorted(data[col].unique()))
                axs[it].grid(axis="y", linestyle="")
                axs[it].set_ylim(data[col].min() - 0.5, data[col].max() + 0.5)
            elif cat_orient == "v":
                axs[it].set_xticks(sorted(data[col].unique()))
                axs[it].grid(axis="x", linestyle="")
                axs[it].set_xlim(data[col].min() - 1, data[col].max() + 1)
            sns.move_legend(axs[it], "upper left", bbox_to_anchor=(1, 1))

    fig.tight_layout()
    plt.show()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
        logger.info(f"custom_hist_multiplot saved to {savepath}")


def custom_joint_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str = "subset",
    title: str | None = None,
    savepath: Path | None = None,
) -> None:
    """
    Generates a joint plot with a scatter plot and marginal histograms
    for the given data. Optionally saves the figure to a file.

    Args:
        data (pd.DataFrame):
            The input data.
        x (str):
            The column name for the x-axis.
        y (str):
            The column name for the y-axis.
        hue (str, optional):
            The column to use for grouping the data. Defaults to "subset".
        title (str | None, optional):
            The title for the plot. Defaults to None.
        savepath (Path | None, optional):
            The file path to save the plot. Defaults to None.

    """
    g = sns.JointGrid(
        data=data, x=x, y=y, hue=hue, height=8, marginal_ticks=True
    )
    g.plot_joint(sns.scatterplot, alpha=0.6)
    g.plot_marginals(
        sns.histplot, common_norm=False, fill=True, kde=True, stat="density"
    )
    if title:
        plt.suptitle(title, y=1.02)
    if savepath:
        g.savefig(savepath, bbox_inches="tight")
        logger.info(f"custom_joint_plot saved to {savepath}")


def custom_box_multiplot(
    data: pd.DataFrame,
    columns: list[str],
    hue: str | None = None,
    title: str | None = None,
    savepath: Path | None = None,
) -> None:
    """
    Generates a multi-panel box plot for the given data. Optionally
    saves the figure to a file.

    Args:
        data (pd.DataFrame):
            The input data.
        columns (list[str]):
            The column names to plot.
        hue (str | None, optional):
            The column to use for grouping the data. Defaults to None.
        title (str | None, optional):
            The title for the plot. Defaults to None.
        savepath (Path | None, optional):
            The file path to save the plot. Defaults to None.

    """
    fig, axs = plt.subplots(len(columns), figsize=(15, 3 * len(columns)))
    if len(columns) == 1:
        axs = [axs]
    if title:
        fig.suptitle(title, fontsize=16)
    for i, col in enumerate(columns):
        sns.boxplot(data=data, x=col, orient="h", ax=axs[i], hue=hue)
        sns.move_legend(axs[i], "upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    plt.show()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
        logger.info(f"custom_box_multiplot saved to {savepath}")


def custom_violin_multiplot(
    data: pd.DataFrame,
    x: str,
    columns: list[str],
    hue: str | None = None,
    hue_order: list | None = None,
    k: float = 0.5,
    savepath: Path | None = None,
) -> None:
    """
    Generates a multi-panel violin plot for the given data. Optionally
    saves the figure to a file.

    Args:
        data (pd.DataFrame):
            The input data.
        x (str):
            The column to use for the x-axis.
        columns (list[str]):
            The column names to plot.
        hue (str | None, optional):
            The column to use for grouping the data. Defaults to None.
        hue_order (list | None, optional):
            The order of the hue categories. Defaults to None.
        k (float, optional):
            The scaling factor for the height of the subplots.
            Defaults to 0.5.
        savepath (Path | None, optional):
            The file path to save the plot. Defaults to None.
    """
    nunique = sum([data[col].nunique() for col in columns])
    ratios = [data[col].nunique() / nunique for col in columns]
    fig, axs = plt.subplots(
        len(columns),
        figsize=(10, k * nunique),
        gridspec_kw={"height_ratios": ratios},
    )
    if len(columns) == 1:
        axs = [axs]
    for it, y in enumerate(columns):
        sns.violinplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            hue_order=hue_order,
            orient="h",
            ax=axs[it],
        )
        axs[it].set_xlabel("")
        axs[it].set_ylabel("")
        axs[it].set_title(y)
        sns.move_legend(axs[it], "upper left", bbox_to_anchor=(1, 1))
    axs[it].set_xlabel(x)
    fig.tight_layout()
    plt.show()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
        logger.info(f"custom_violin_multiplot saved to {savepath}")
