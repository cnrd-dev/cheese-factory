"""
Custom library for data analysis.

Functions for descriptive statistics and regression analysis.
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


def plot_graphs(x1: pd.Series, x2: pd.Series, df: pd.DataFrame, feature: str, title: str) -> None:
    """Generate histogram and box plots to compare APC off versus APC on.

    Args:
        x1 (pd.Series): Series for APC off data.
        x2 (pd.Series): Series for APC on data.
        df (pd.DataFrame): Dataframe containing the data.
        feature (str): Column name of the feature to use in the dataframe.
        title (str): Title of the plot.
    """
    fig = go.Figure()
    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(go.Histogram(x=x1, name="APC OFF", xbins=dict(size=hist_bin_width_fd(df[feature])), histnorm="probability", marker=dict(color="rgba(198,12,48,0.5)")), row=1, col=1)
    fig.add_trace(go.Histogram(x=x2, name="APC ON", xbins=dict(size=hist_bin_width_fd(df[feature])), histnorm="probability", marker=dict(color="rgba(0,39,118,0.5)")), row=1, col=1)

    fig.add_trace(go.Box(y=x1, name="APC OFF", boxmean="sd", fillcolor="rgba(198,12,48,0.5)", marker=dict(color="rgba(198,12,48,0.5)")), row=1, col=2)
    fig.add_trace(go.Box(y=x2, name="APC ON", boxmean="sd", fillcolor="rgba(0,39,118,0.5)", marker=dict(color="rgba(0,39,118,0.5)")), row=1, col=2)

    fig["layout"].update(
        title="<b>" + title + "</b><br>Date range: " + str(min(x1.index)) + " to " + str(max(x1.index)) + "</i>",
        font=dict(size=9),
        margin=dict(l=60, r=60, t=60, b=60),
        annotations=[
            dict(x=0, y=-0.2, showarrow=False, text="The mean is represented by the dashed horizontal line and the standard deviation by the dashed diamond shape.", xref="paper", yref="paper")
        ],
        barmode="overlay",
        showlegend=False,
    )
    fig.show(renderer="notebook")


def plot_timeseries(df: pd.DataFrame, y_traces: list, title: str, x_trace: str = "", use_index: bool = True) -> None:
    """Plot timeseries data from dataframe using plotly library.

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
