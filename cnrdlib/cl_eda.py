"""
Custom library for Exploratory Data Analysis.

Functions for descriptive statistics and plots.
"""

from typing import Tuple
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


def hist_bin_width_fd(x: pd.Series) -> float:
    """Create bin widths for histograms based on the Freedman-Diaconis rule.

    Args:
        x (pd.Series): Series of data to use to generate bin widths.

    Returns:
        float: Number that specifies the bin widths.
    """
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    h = 2.0 * iqr * x.size ** (-1.0 / 3.0)
    if (x.max() - x.min()) / h > 1e8 * x.size:
        warnings.warn("Bin width estimated with the Freedman-Diaconis rule is very small" " (= {})".format(h), RuntimeWarning, stacklevel=2)
    return h


def plot_hist(cols: list, data: pd.DataFrame) -> None:
    """Plot multiple histograms.

    Args:
        cols (list): List of column names to plot.
        data (pd.DataFrame): Dataframe containing the data.
    """
    collen = len(cols)
    rows = math.ceil(collen / 3)
    rownum = 1
    colnum = 1
    fig = make_subplots(rows=rows, cols=3, subplot_titles=(cols))

    for col in cols:
        fig.add_trace(go.Histogram(x=data[col], xbins=dict(size=hist_bin_width_fd(data[col]))), row=rownum, col=colnum)

        colnum = colnum + 1
        if colnum > 3:
            rownum = rownum + 1
            colnum = 1
    fig.update_layout(
        height=300 * rows,
        width=1300,
        bargap=0.05,
        showlegend=False,
    )
    fig.show()


def plot_timeseries_plotly(df: pd.DataFrame, y_traces: list, title: str = "", x_trace: str = "", use_index: bool = True) -> None:
    """Plot timeseries data from dataframe using plotly library for interactive graphs.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        y_traces (list): List of columns to include for the y-axis.
        title (str): Title for the plot.
        x_trace (str, optional): Name of column to use as x-axis. Defaults to "".
        use_index (bool, optional): Specificy as False to use column in x_trace. Defaults to True.
    """
    fig = go.Figure()

    if use_index == True:
        X_trace = df.index
    else:
        X_trace = df[x_trace]

    for y_trace in y_traces:
        fig.add_trace(
            go.Scatter(
                x=X_trace,
                y=df[y_trace],
                name=y_trace,
            )
        )

    fig["layout"].update(title="<b>" + title + "</b>")
    fig.show(renderer="notebook")


def plot_timeseries_static(df: pd.DataFrame, y1: str, y2: str, title: str = "") -> None:
    """Plot timeseries data from dataframe for static graphs.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        y1 (str): Label for the primary y-axis.
        y2 (str): Label for the secondary y-axis.
        title (str): Title for the plot.
    """
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(df.index, df[y1])
    ax2.plot(df.index, df[y2], color="red")

    ax1.set_ylabel(y1)
    ax2.set_ylabel(y2)

    fig.suptitle(title)

    plt.show()


def plot_anomaly(seq: int, timesteps: int, xtest: np.array, xtestpred: np.array) -> None:
    """Plot timeseries sequence with reconstruction and error.

    Args:
        seq (int): Sequence to plot.
        timesteps (int): Timesteps used in the model.
        xtest (nd.array): Test
    """
    plt.plot(xtest[seq], "b")
    plt.plot(xtestpred[seq], "r")
    plt.fill_between(np.arange(timesteps), np.squeeze(xtestpred[seq]), np.squeeze(xtest[seq]), color="lightcoral")
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.show()
