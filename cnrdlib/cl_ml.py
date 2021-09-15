"""
Custom library for Machine Learning.

Functions for descriptive regression analysis.
"""

from typing import Tuple
from statsmodels.graphics.gofplots import ProbPlot
import math
import sklearn.metrics as sklm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import math

warnings.filterwarnings("ignore")


def regression_metrics(y_true, y_predicted, n_parameters):
    """Calculate regression metrics.

    Args:
        y_true ([type]): [description]
        y_predicted ([type]): [description]
        n_parameters ([type]): [description]
    """

    r2 = sklm.r2_score(y_true, y_predicted)
    r2_adj = r2 - (n_parameters - 1) / (y_true.shape[0] - n_parameters) * (1 - r2)

    print(f"Root Mean Square Error = {math.sqrt(sklm.mean_squared_error(y_true, y_predicted)):0.2f}")
    print(f"Mean Absolute Error    = {sklm.mean_absolute_error(y_true, y_predicted):0.2f}")
    print(f"Median Absolute Error  = {sklm.median_absolute_error(y_true, y_predicted):0.2f}")
    print(f"R^2                    = {r2:0.4f}")
    print(f"Adjusted R^2           = {r2_adj:0.4f}")


def regression_eval_metrics(x_true: np.ndarray, y_true: np.ndarray, y_predicted: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate regression evaluation metrics.

    Args:
        x_true (np.ndarray): Known x-values.
        y_true (np.ndarray): Known y-values.
        y_predicted (np.ndarray): Predicted y-values.

    Returns:
        residuals (np.ndarray): Residuals from y_true and y_predicted.
        studentized_residuals (np.ndarray): Standardized residuals from residuals.
        cooks_distance (np.ndarray): Cooks distance values from residuals and hat_diag.
        hat_diag (np.ndarray): Diagonal array of x_true.
    """
    p = np.size(x_true, 1)
    residuals = np.subtract(y_true, y_predicted)
    X = x_true
    hat = X.dot(np.linalg.inv(X.T.dot(X)).dot(X.T))
    hat_diag = np.diag(hat)
    MSE = sklm.mean_squared_error(y_true, y_predicted)
    studentized_residuals = residuals / np.sqrt(MSE * (1 - hat_diag))
    cooks_distance = (residuals ** 2 / (p * MSE)) * (hat_diag / (1 - hat_diag) ** 2)
    return residuals, studentized_residuals, cooks_distance, hat_diag


def diagnostic_plots(x_true: np.ndarray, y_true: np.ndarray, y_predicted: np.ndarray) -> None:
    """Generate diagnostic plots for regression evaluation.

    Source: Emre @ https://emredjan.github.io/blog/2017/07/11/emulating-r-plots-in-python/

    Args:
        x_true (np.ndarray): Known x-values.
        y_true (np.ndarray): Known y-values.
        y_predicted (np.ndarray): Predicted y-values.
    """
    residuals, studentized_residuals, cooks_distance, hat_diag = regression_eval_metrics(x_true, y_true, y_predicted)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))
    plt.tight_layout(pad=5, w_pad=5, h_pad=5)

    # 1. residual plot
    sns.residplot(x=y_predicted, y=residuals, lowess=True, scatter_kws={"alpha": 0.5}, line_kws={"color": "red", "lw": 1, "alpha": 0.8}, ax=axs[0, 0])
    axs[0, 0].set_title("Residuals vs Fitted")
    axs[0, 0].set_xlabel("Fitted values")
    axs[0, 0].set_ylabel("Residuals")

    # 2. qq plot
    qq = ProbPlot(studentized_residuals)
    qq.qqplot(line="45", alpha=0.5, color="#2578B2", lw=0.5, ax=axs[0, 1])
    axs[0, 1].set_title("Normal Q-Q")
    axs[0, 1].set_xlabel("Theoretical Quantiles")
    axs[0, 1].set_ylabel("Standardized Residuals")

    # 3. scale-location plot
    studentized_residuals_abs_sqrt = np.sqrt(np.abs(studentized_residuals))
    axs[1, 0].scatter(y_predicted, studentized_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(
        y_predicted,
        studentized_residuals_abs_sqrt,
        scatter=False,
        ci=False,
        lowess=True,
        line_kws={"color": "red", "lw": 1, "alpha": 0.8},
        ax=axs[1, 0],
    )
    axs[1, 0].set_title("Scale-Location")
    axs[1, 0].set_xlabel("Fitted values")
    axs[1, 0].set_ylabel("$\sqrt{|Standardised Residuals|}$")

    # 4. leverage plot
    axs[1, 1].scatter(hat_diag, studentized_residuals, alpha=0.5)
    sns.regplot(hat_diag, studentized_residuals, scatter=False, ci=False, lowess=True, line_kws={"color": "red", "lw": 1, "alpha": 0.8}, ax=axs[1, 1])
    axs[1, 1].set_xlim(min(hat_diag), max(hat_diag))
    axs[1, 1].set_ylim(min(studentized_residuals), max(studentized_residuals))
    axs[1, 1].set_title("Residuals vs Leverage")
    axs[1, 1].set_xlabel("Leverage")
    axs[1, 1].set_ylabel("Standardised Residuals")

    # annotations
    leverage_top_3 = np.flip(np.argsort(cooks_distance), 0)[:3]
    for i in leverage_top_3:
        axs[1, 1].annotate(i, xy=(hat_diag[i], studentized_residuals[i]))

    def graph(formula, x_range, label=None):
        x = x_range
        y = formula(x)
        axs[1, 1].plot(x, y, label=label, lw=1, ls="--", color="red")

    p = np.size(x_true, 1)  # number of model parameters

    graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), np.linspace(0.001, max(hat_diag), 50), "Cook's distance")
    graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), np.linspace(0.001, max(hat_diag), 50))
    axs[1, 1].legend(loc="upper right")
