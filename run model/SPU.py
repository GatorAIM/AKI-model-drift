import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import joblib
import json
import os
import random

import sklearn.metrics
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, ParameterGrid
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score,make_scorer, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier
from sklearn.utils import resample
from scipy.stats import randint, uniform, loguniform


def load_aggregated_data(data_by_year_totall, data_by_year_training, cutoff_year, update_year):
    """
    Load and aggregate training data for a specified cutoff year and update interval.

    Parameters:
    - data_by_year_totall (pd.DataFrame): Fixed portion of the data to include in aggregation.
    - data_by_year_training (dict): Dictionary of training data by year.
    - cutoff_year (int): The year up to which data should be aggregated.
    - update_year (int): The interval for dynamically updating the aggregated data.

    Returns:
    - X_train (pd.DataFrame): Features for training.
    - y_train (pd.Series): Target variable for training.
    """
    # Aggregate data: fixed portion + dynamically updated portion
    aggregated_data = pd.concat(
        [data_by_year_totall] +
        [
            data_by_year_training[year] 
            for year in range(2012, cutoff_year + 1, update_year)  # Dynamically update up to cutoff_year
            if year in data_by_year_training  # Ensure the year exists in the data
        ],
        axis=0
    )

    # Clean the data by dropping unnecessary columns
    aggregated_data_cleaned = aggregated_data.drop(
        columns=['ENCOUNTERID', 'SINCE_ADMIT', 'PATID', 'ADMIT_DATE', 'BCCOVID'], 
        errors='ignore'
    )

    # Prepare training data
    y_train = aggregated_data_cleaned['FLAG']
    X_train = aggregated_data_cleaned.drop(columns=['FLAG'], errors='ignore')

    return X_train, y_train

def tune_model_with_random_search(param_distributions, use_scale_pos_weight, X_train, y_train):
    """
    Load and use a fixed set of hyperparameters to initialize XGBClassifier.

    Parameters:
    - param_distributions (dict): Fixed hyperparameter set.
    - use_scale_pos_weight (bool): Whether to calculate and use scale_pos_weight.
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.

    Returns:
    - best_model (XGBClassifier): XGBClassifier initialized with fixed parameters.
    - best_params (dict): The fixed parameters.
    """

    pos_weight = None
    if use_scale_pos_weight:
        pos_weight = sum(y_train == 0) / sum(y_train == 1)

    best_params = {
        'n_estimators': param_distributions['n_estimators'][0],  
        'max_depth': param_distributions['max_depth'][0],  
        'learning_rate': param_distributions['learning_rate'][0],  
        'subsample': param_distributions['subsample'][0],  
        'colsample_bytree': param_distributions['colsample_bytree'][0]
    }

    print(f"⚡ Using fixed hyperparameters: {best_params}")

    best_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=pos_weight if use_scale_pos_weight else None,
        **best_params  
    )

    best_model.fit(X_train, y_train)

    return best_model, best_params


def evaluate_model_performance_with_cv(data_by_year_test, test_year, best_model):
    """
    Evaluate the performance of the best model on a specific test year's data using cross-validation.

    Parameters:
    - data_by_year_test (dict): Dictionary containing test data by year.
    - test_year (int): The year for which to evaluate the model.
    - best_model (sklearn estimator): The trained model to evaluate.
    - n_splits (int): Number of folds for cross-validation.

    Returns:
    - metrics_mean (dict): Mean of metrics (AUC, PRAUC, F1, Precision, Recall).
    - metrics_variance (dict): Variance of metrics (AUC, PRAUC, F1, Precision, Recall).
    """
    # Extract test data for the specified year
    test_data = data_by_year_test[test_year]
    test_y = test_data['FLAG']
    test_X = test_data.drop(columns=['ENCOUNTERID', 'SINCE_ADMIT', 'PATID', 'ADMIT_DATE', 'BCCOVID', 'FLAG'], errors='ignore')

    # Model predictions
    test_pred_prob = best_model.predict_proba(test_X)[:, 1]
    test_pred = best_model.predict(test_X)
    
    auc = roc_auc_score(test_y, test_pred_prob)
    prauc = average_precision_score(test_y, test_pred_prob)
    f1 = f1_score(test_y, test_pred)
    precision_value = precision_score(test_y, test_pred)
    recall_value = recall_score(test_y, test_pred)
    
    # Compile metrics
    metrics = {
        'AUC': auc,
        'PRAUC': prauc,
        'F1': f1,
        'Precision': precision_value,
        'Recall': recall_value
    }

    return metrics

def evaluate_model_with_checks(test_data_dict, test_year, best_model, category, class_type):
    """
    Evaluate model performance with checks for data availability and class imbalance using cross-validation.
 
    Parameters:
    - test_data_dict (dict): Dictionary containing test data.
    - test_year (int): The year to test.
    - best_model (sklearn estimator): The trained model to evaluate.
    - category (str): The category of the data.
    - class_type (str): The class type (e.g., '1', '0').
    - n_splits (int): Number of folds for cross-validation.

    Returns:
    - metrics_mean (dict): Mean of metrics (AUC, PRAUC, F1, Precision, Recall).
    - metrics_variance (dict): Variance of metrics (AUC, PRAUC, F1, Precision, Recall).
    """
    # Extract test data
    test_data = test_data_dict[test_year]
    test_y = test_data['FLAG']
    test_X = test_data.drop(
        columns=['ENCOUNTERID', 'SINCE_ADMIT', 'PATID', 'ADMIT_DATE', 'BCCOVID', 'FLAG'],
        errors='ignore'
    )

    # Check if test_y contains at least two different classes
    if len(np.unique(test_y)) <= 1:
        print(f"Only one class present in y_true for category {category}, class_type {class_type}, year {test_year}. Filling with NaN.")
        return {
            'AUC': np.nan,
            'PRAUC': np.nan,
            'F1-Score': np.nan,
            'Precision': np.nan,
            'Recall': np.nan
        }

    try:
        # Model predictions
        test_pred_prob = best_model.predict_proba(test_X)[:, 1]
        test_pred = best_model.predict(test_X)

        # Compute evaluation metrics
        auc = roc_auc_score(test_y, test_pred_prob)
        prauc = average_precision_score(test_y, test_pred_prob)
        f1 = f1_score(test_y, test_pred)
        precision_value = precision_score(test_y, test_pred)
        recall_value = recall_score(test_y, test_pred)
        # Compile metrics
        metrics = {
            'AUC': auc,
            'PRAUC': prauc,
            'F1-Score': f1,
            'Precision': precision_value,
            'Recall': recall_value
        }
        return metrics
    except Exception as e:
        print(f"Error processing fold for category {category}, class_type {class_type}, year {test_year}: {e}")
        return {
            'AUC': np.nan,
            'PRAUC': np.nan,
            'F1-Score': np.nan,
            'Precision': np.nan,
            'Recall': np.nan
        }
