# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2023/05/09
# Forum: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2023 All Rights Reserved.
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxnetSurvivalAnalysis

from onekey_algo.custom.components.comp1 import normalize_df

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FitFailedWarning)

from onekey_algo import init_CN, get_param_in_cwd

COEF_THRESHOLD = 1e-6


def lasso_cox_cv(X_data, y_data, alpha_logmin=-3, points=50, cv: int = 10, save_dir: str = 'img',
                 prefix: str = '', l1_ratio=1, random_state=0, max_iter: int = 100, n_highlight: int = 0,
                 norm_X: bool = True, **kwargs):
    """

    Args:
        X_data: 训练数据
        y_data: 监督数据
        alpha_logmin: alpha的log最小值
        points: 打印多少个点。默认50
        cv: 交叉验证次数。
        save_dir: 保存目录
        prefix: ''
        l1_ratio:
        max_iter:
        n_highlight:
        random_state: 0
        norm_X: bool, 是否标准化X_data
        **kwargs: 其他用于打印控制的参数。

    Returns:
        none zero features with coefficient

    """
    precision = get_param_in_cwd('display.precision', 3)
    if norm_X:
        # display(X_data.describe())
        X_data = normalize_df(X_data)
        X_data = X_data.dropna(axis=1)
        # display(X_data.describe())
    # 每个特征值随lambda的变化
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    init_CN()
    if points != 50 or cv != 10:
        print(f"Points: {points}, CV: {cv}")
    coxnet_pipe = make_pipeline(  # StandardScaler(),
        CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=10 ** alpha_logmin,
                               n_alphas=points, max_iter=max_iter))
    coxnet_pipe.fit(X_data, y_data)

    cox_lasso = coxnet_pipe.named_steps["coxnetsurvivalanalysis"]
    coefficients_lasso = pd.DataFrame(
        cox_lasso.coef_,
        index=X_data.columns,
        columns=np.round(cox_lasso.alphas_, 5)
    )
    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    # estimated_alphas = alphas
    cv_func = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    gcv = GridSearchCV(
        make_pipeline(  # StandardScaler(),
            CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=10 ** alpha_logmin,
                                   n_alphas=points, max_iter=max_iter)),
        param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
        cv=cv_func,
        error_score=0.5).fit(X_data, y_data)

    cv_results = pd.DataFrame(gcv.cv_results_)

    # 绘制Lasso路径
    bst_alpha = gcv.best_params_["coxnetsurvivalanalysis__alphas"][0]
    lambda_info = f"(λ={{l:.{precision}f}})"
    lambda_info = lambda_info.format(l=bst_alpha)

    def plot_coefficients(coefs, n_highlight):
        alphas = coefs.columns
        for row in coefs.itertuples():
            plt.semilogx(alphas, row[1:], "-", label=row.Index)

        alpha_min = alphas.min()
        top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
        for name in top_coefs.index:
            coef = coefs.loc[name, alpha_min]
            plt.text(
                alpha_min, coef, name + "   ",
                horizontalalignment="right",
                verticalalignment="center"
            )

        #     ax.yaxis.set_label_position("right")
        #     ax.yaxis.tick_right()
        #     ax.grid(True)
        plt.xlabel(f'Lambda{lambda_info}')
        plt.ylabel("coefficient")
        plt.axvline(bst_alpha, color='black', ls="--")
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f'{prefix}feature_lasso.svg'), bbox_inches='tight')
            plt.show()

    plot_coefficients(coefficients_lasso, n_highlight=n_highlight)
    # 绘制MSE
    alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score

    # fig, ax = plt.subplots()
    plt.plot(alphas, mean)
    plt.fill_between(alphas, mean - std, mean + std, alpha=.15)
    plt.xscale("log")
    plt.ylabel("concordance index")
    plt.xlabel(f'Lambda{lambda_info}')
    plt.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], color='black', ls="--")
    plt.axhline(0.5, color="grey", linestyle="--")
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{prefix}feature_cindex.svg'), bbox_inches='tight')
        plt.show()

    # 绘制参数Weights
    best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
    best_coefs = pd.DataFrame(
        best_model.coef_,
        index=X_data.columns,
        columns=["coefficient"]
    )

    non_zero_coefs = best_coefs.query("coefficient != 0")
    coef_order = non_zero_coefs.sort_values("coefficient").index

    # _, ax = plt.subplots()
    non_zero_coefs.loc[coef_order].plot.barh(legend=False)
    plt.xlabel("coefficient")
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{prefix}feature_weights.svg'), bbox_inches='tight')
        plt.show()

    # 输出Lasso公式
    selected_features = []
    feat_coef = [(feat_name, coef) for feat_name, coef in zip(non_zero_coefs.loc[coef_order].index,
                                                              non_zero_coefs.loc[coef_order]['coefficient'])
                 if COEF_THRESHOLD is None or abs(coef) > COEF_THRESHOLD]
    selected_features.append([feat for feat, _ in feat_coef])
    formula = ' '.join([f"{coef:+.6f} * {feat_name}" for feat_name, coef in feat_coef])
    if save_dir is not None:
        print(f"Survive = {formula}")
    return non_zero_coefs.loc[coef_order]


def get_x_y_survival(dataset, event_col, duration_col, val_outcome=1):
    if event_col is None or duration_col is None:
        y = None
        x_frame = dataset
    else:
        y = np.empty(dtype=[(event_col, bool), (duration_col, np.float64)],
                     shape=dataset.shape[0])
        y[event_col] = (dataset[event_col] == val_outcome).values
        y[duration_col] = dataset[duration_col].values

        x_frame = dataset[[c for c in dataset.columns if c not in [duration_col, event_col, 'ID', 'group']]]

    return x_frame, y


def get_prediction(model: CoxPHFitter, data, ID=None, **kwargs):
    hr = model.predict_partial_hazard(data)
    expectation = model.predict_expectation(data)

    predictions = pd.concat([hr, expectation], axis=1)
    predictions.columns = ['HR', 'expectation']
    if ID is not None:
        predictions = pd.concat([ID, hr, expectation], axis=1)
        predictions.columns = ['ID', 'HR', 'expectation']
    else:
        predictions = pd.concat([hr, expectation], axis=1)
        predictions.columns = ['HR', 'expectation']
    return predictions
