#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 08/12/2024 20:01
@author: Hendrik

Analyse Covid vaccination survey responses
"""
import numpy as np
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import warnings
import re
import matplotlib.pyplot as plt


def load_data(file_path):

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=re.escape('openpyxl.styles.stylesheet'))
        content = pd.read_excel(file_path)

    ### DATA CLEANUP ###
    # Replace "array" in age (Q3) with nan
    content.loc[content['Q3'] == 'Array', 'Q3'] = np.nan
    content.loc[content['Q3'] == '＞40', 'Q3'] = 41

    question_map = pd.DataFrame(content.iloc[0])
    content = content.drop(0, axis=0)

    return content, question_map


def factorize_answers(df_label, df_value, column):

    df_value_local = df_value.copy()

    # Factorize column in df_label and df_value
    lab_label, uniques_label = pd.factorize(df_label[column])
    lab_value, uniques_value = pd.factorize(df_value[column])

    if not all(np.equal(lab_label, lab_value)):
        raise ValueError('Factorization failed.')

    # Add factorized column to df_value
    df_value_local[f'{column}_fac'] = lab_value

    # Create Dataframe that maps factorized labels to values
    mapper = pd.DataFrame(data=[uniques_label, uniques_value], index=['labels', 'values']).T

    return df_value_local, mapper


def sort_labels(val1, val2):
    mapping = dict(zip(val1, val2))
    # Return reordered labels, filter out nans
    return [x for x in sorted(mapping, key=mapping.get) if not (not isinstance(x, str) and np.isnan(x))]


def run_glm(df, dep, indep, exclude_indep=None):

    indep_local = indep.copy()

    if exclude_indep is not None:
        for ex in exclude_indep:
            if ex in indep_local:
                indep_local.remove(ex)

    # Dependent variable --> which variable do I want to test?
    y = df[dep]

    # Independent variables --> which factors might influence the dependent variable?
    x = df[indep_local]
    x = sm.add_constant(x)

    # Get a nan mask for X and Y
    y_nan = pd.isna(y)
    x_nan = pd.isna(x).sum(axis=1) > 0
    nan_mask = ~(y_nan + x_nan)

    model = sm.OLS(endog=y.astype(float)[nan_mask], exog=x.astype(float)[nan_mask])
    results = model.fit()

    return results


def plot_question_distribution(df_lab, df_val, cat1, cat2, ax=None, question_map=None, save_fig=False):

    # Order categorical data based on their integer values
    x_order = sort_labels(df_lab[cat1], df_val[cat1])
    hue_order = sort_labels(df_lab[cat2], df_val[cat2])

    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    sns.countplot(data=df_lab, x=cat1, hue=cat2, ax=ax, order=x_order, hue_order=hue_order)

    if question_map is not None:
        q_map = questions[0].to_dict()
        ax.set_xlabel(q_map[cat1])
        ax.get_legend().set_title(q_map[cat2])

        sns.move_legend(
            ax, "lower center",
            bbox_to_anchor=(.5, 1), ncol=len(hue_order), frameon=False,
        )

    if save_fig:
        fig.set_size_inches(24.5, 12.5)
        plt.tight_layout()
        plt.savefig(f'img\\{cat2}_{cat1}_bar.png')
        plt.close()



if __name__ == '__main__':

    # Plot the distribution of two questions/categories
    save_fig = False

    data_label, questions = load_data(file_path=r'QualtrixLabels.xlsx')
    data_value, questions = load_data(file_path=r'QualtrixValues.xlsx')

    source_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q9_fac']

    # Factorize question 9 (vaccine availability)
    data_value, q9_map = factorize_answers(df_label=data_label, df_value=data_value, column='Q9')

    # Export cleaned data tables as CSV
    data_value.to_csv('data_value.csv')
    data_label.to_csv('data_label.csv')

    # Influence on vaccination status
    result = run_glm(df=data_value, dep='Q15', indep=source_questions, exclude_indep=['Q9_fac'])
    plot_question_distribution(df_lab=data_label, df_val=data_value, cat1='Q14', cat2='Q15', question_map=questions, save_fig=save_fig)

    # Influence on future vaccination
    result = run_glm(df=data_value, dep='Q21', indep=source_questions)
    plot_question_distribution(df_lab=data_label, df_val=data_value, cat1='Q5', cat2='Q21', question_map=questions, save_fig=save_fig)
    plot_question_distribution(df_lab=data_label, df_val=data_value, cat1='Q7', cat2='Q21', question_map=questions, save_fig=save_fig)
    plot_question_distribution(df_lab=data_label, df_val=data_value, cat1='Q11', cat2='Q21', question_map=questions, save_fig=save_fig)
    plot_question_distribution(df_lab=data_label, df_val=data_value, cat1='Q14', cat2='Q21', question_map=questions, save_fig=save_fig)

    # Influence on vaccination timing
    result = run_glm(df=data_value, dep='Q17', indep=source_questions)
    plot_question_distribution(df_lab=data_label, df_val=data_value, cat1='Q2', cat2='Q17', question_map=questions, save_fig=save_fig)
    plot_question_distribution(df_lab=data_label, df_val=data_value, cat1='Q6', cat2='Q17', question_map=questions, save_fig=save_fig)
    plot_question_distribution(df_lab=data_label, df_val=data_value, cat1='Q10', cat2='Q17', question_map=questions, save_fig=save_fig)
    plot_question_distribution(df_lab=data_label, df_val=data_value, cat1='Q13', cat2='Q17', question_map=questions, save_fig=save_fig)

    # Influence on vaccination speed
    result = run_glm(df=data_value, dep='Q20', indep=source_questions)
    plot_question_distribution(df_lab=data_label, df_val=data_value, cat1='Q5', cat2='Q20', question_map=questions, save_fig=save_fig)
    plot_question_distribution(df_lab=data_label, df_val=data_value, cat1='Q10', cat2='Q20', question_map=questions, save_fig=save_fig)
    plot_question_distribution(df_lab=data_label, df_val=data_value, cat1='Q11', cat2='Q20', question_map=questions, save_fig=save_fig)
    plot_question_distribution(df_lab=data_label, df_val=data_value, cat1='Q12', cat2='Q20', question_map=questions, save_fig=save_fig)

