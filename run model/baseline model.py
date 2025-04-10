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
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset

def load_data_by_year(save_path):
    """
    Load training and test datasets by year from the specified path.

    Parameters:
    save_path (str): The directory path containing the training and test dataset files.

    Returns:
    tuple: A tuple containing two dictionaries:
        - data_by_year_training: A dictionary where keys are years and values are training data DataFrames.
        - data_by_year_test: A dictionary where keys are years and values are test data DataFrames.
    """
    # Initialize dictionaries to store datasets by year
    data_by_year_training = {}
    data_by_year_test = {}

    # Iterate through files in the specified save path
    for file_name in os.listdir(save_path):
        # Check if the file corresponds to training data
        if file_name.startswith("training_data_"):
            year = int(file_name.split("_")[2].split(".")[0])  # Extract year from file name
            file_path = os.path.join(save_path, file_name)

            # Load CSV into the training dictionary
            data_by_year_training[year] = pd.read_csv(file_path)
            #print(f"Training data for year {year} loaded from {file_path}.")

        # Check if the file corresponds to test data
        elif file_name.startswith("test_data_"):
            year = int(file_name.split("_")[2].split(".")[0])  # Extract year from file name
            file_path = os.path.join(save_path, file_name)

            # Load CSV into the test dictionary
            data_by_year_test[year] = pd.read_csv(file_path)
            #print(f"Test data for year {year} loaded from {file_path}.")

    print("All training and test datasets have been loaded.")
    return data_by_year_training, data_by_year_test


def drop_high_correlation_variables(file_path, data_by_year_training, data_by_year_test):
    """
    Drop high correlation variables from training and test datasets based on a provided variable list file.

    Args:
        file_path (str): Path to the text file containing variable names to drop (one per line).
        data_by_year_training (dict): Dictionary of training datasets indexed by year.
        data_by_year_test (dict): Dictionary of test datasets indexed by year.

    Returns:
        tuple: (updated data_by_year_training, updated data_by_year_test)
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            to_drop = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(to_drop)} variables to drop.")
    else:
        print(f"File {file_path} not found.")
        to_drop = []

    for year in data_by_year_training.keys():
        drop_columns_train = [col for col in to_drop if col in data_by_year_training[year].columns]
        drop_columns_test = [col for col in to_drop if col in data_by_year_test[year].columns]

        data_by_year_training[year] = data_by_year_training[year].drop(columns=drop_columns_train)
        data_by_year_test[year] = data_by_year_test[year].drop(columns=drop_columns_test)

        print(f"Year {year}: Dropped {len(drop_columns_train)} columns from training set, {len(drop_columns_test)} from test set.")

    return data_by_year_training, data_by_year_test

def split_data_by_category(categories, data_by_year, data_type="test"):
    """
    Split datasets by category and label (1 or 0) for each year.

    Parameters:
    categories (list): List of categories to split data on.
    data_by_year (dict): Dictionary containing datasets by year.
    data_type (str): Either "test" or "training", used for dynamic naming of globals.

    Returns:
    None: Assigns split dictionaries to global variables dynamically.
    """
    # Iterate through categories
    for category in categories:
        class1_dict = {}
        class0_dict = {}

        for year, data in data_by_year.items():
            try:
                # Step 1: Convert data types
                if 'PATID' in data.columns:
                    data['PATID'] = data['PATID'].astype(float)
                if 'ENCOUNTERID' in data.columns:
                    data['ENCOUNTERID'] = data['ENCOUNTERID'].astype(float)
                if 'ADMIT_DATE' in data.columns:
                    data['ADMIT_DATE'] = pd.to_datetime(data['ADMIT_DATE'], errors='coerce')

                # Step 2: Check if category exists
                if category not in data.columns:
                    raise KeyError(f"{category} column does not exist")

                # Step 3: Split data by category
                class1_dict[year] = data[data[category] == 1].copy()
                class0_dict[year] = data[data[category] == 0].copy()

                # Print split results
                #print(f"Year {year} Category {category}: Split completed - {category}=1: {len(class1_dict[year])}, {category}=0: {len(class0_dict[year])}")

            except KeyError as e:
                print(f"Year {year} missing required column: {e}")
            except Exception as e:
                print(f"Error processing year {year} category {category}: {e}")

        # Assign split data to global variables dynamically
        globals()[f"data_by_year_{data_type}_{category}_1"] = class1_dict
        globals()[f"data_by_year_{data_type}_{category}_0"] = class0_dict
        
def drop_columns_from_data(columns_to_drop, data_by_year):
    """
    Drop specified columns from datasets by year.

    Parameters:
    columns_to_drop (list): List of column names to drop.
    data_by_year (dict): Dictionary containing datasets by year.

    Returns:
    dict: Updated dictionary with specified columns dropped.
    """
    for year, df in data_by_year.items():
        # Identify columns to drop that exist in the DataFrame
        cols_in_df = [col for col in columns_to_drop if col in df.columns]
        if cols_in_df:
            data_by_year[year] = df.drop(columns=cols_in_df)
            #print(f"Dropped columns {cols_in_df} from data for year {year}.")
        else:
            print(f"No matching columns to drop in data for year {year}.")

    return data_by_year


def drop_columns_by_category_and_class(categories, columns_to_drop, data_type="test"):
    """
    Drop specified columns for each category and class (1 or 0) in the global data dictionaries.

    Parameters:
    categories (list): List of categories to process.
    columns_to_drop (list): List of column names to drop.
    data_type (str): Either "test" or "training", used for dynamic naming of globals.

    Returns:
    None: Updates global dictionaries dynamically.
    """
    for category in categories:
        for class_type in ['1', '0']:  # Process data for class 1 and 0
            # Dynamically load the corresponding dictionary
            data_dict = globals().get(f"data_by_year_{data_type}_{category}_{class_type}", {})

            if not data_dict:
                print(f"No {data_type} data found for category {category} class {class_type}.")
                continue

            # Iterate through the dictionary and drop specified columns
            for year, df in data_dict.items():
                # Identify columns to drop that exist in the DataFrame
                cols_in_df = [col for col in columns_to_drop if col in df.columns]
                if cols_in_df:
                    data_dict[year] = df.drop(columns=cols_in_df)
                    #print(f"Dropped columns {cols_in_df} from {data_type} data for category {category} class {class_type}, year {year}.")
                else:
                    print(f"No matching columns to drop in {data_type} data for category {category} class {class_type}, year {year}.")

            # Update the global dictionary
            globals()[f"data_by_year_{data_type}_{category}_{class_type}"] = data_dict
            
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

    

def tune_model_with_random_search(param_distributions, use_scale_pos_weight, X_train, y_train, n_iter1):
    """
    Perform hyperparameter tuning using RandomizedSearchCV on an XGBClassifier.

    Parameters:
    - param_distributions (dict): Hyperparameter search space.
    - use_scale_pos_weight (bool): Whether to calculate and use scale_pos_weight.
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.

    Returns:
    - best_model (XGBClassifier): The best model found by RandomizedSearchCV.
    - best_params (dict): The best parameters found by RandomizedSearchCV.
    """
    global all_hyperparameter_results
    
    # Calculate scale_pos_weight if specified
    pos_weight = None
    if use_scale_pos_weight:
        pos_weight = sum(y_train == 0) / sum(y_train == 1)

    # Initialize the base model
    base_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,  # Use all available cores
        scale_pos_weight=pos_weight if use_scale_pos_weight else None
    )
    prauc_scorer = make_scorer(average_precision_score, needs_proba=True)

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter1,  # Number of parameter settings sampled
        scoring=prauc_scorer,
        cv=5,  # 5-fold cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1  # Use all available cores for RandomizedSearchCV
    )

    # Fit the model to the training data
    random_search.fit(X_train, y_train)
    
    results_df = pd.DataFrame(random_search.cv_results_)
    for _, row in results_df.iterrows():
        all_hyperparameter_results.append({
            'params': row['params'],
            'mean_test_score': row['mean_test_score'],
            'std_test_score': row['std_test_score']
        })


    # Return both the best model and the best parameters
    return random_search.best_estimator_, random_search.best_params_



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


def evaluate_model_performance_with_cv_LR(data_by_year_test, test_year, best_model_LR):
    test_data = data_by_year_test[test_year]
    test_y = test_data['FLAG']
    test_X = test_data.drop(columns=['ENCOUNTERID', 'SINCE_ADMIT', 'PATID', 'ADMIT_DATE', 'BCCOVID', 'FLAG'], errors='ignore')

    test_pred_prob = best_model_LR.predict_proba(test_X)[:, 1]
    test_pred = best_model_LR.predict(test_X)
    
    auc = roc_auc_score(test_y, test_pred_prob)
    prauc = average_precision_score(test_y, test_pred_prob)
    f1 = f1_score(test_y, test_pred)
    precision_value = precision_score(test_y, test_pred)
    recall_value = recall_score(test_y, test_pred)

    return {
        'AUC': auc,
        'PRAUC': prauc,
        'F1': f1,
        'Precision': precision_value,
        'Recall': recall_value
    }

    
def tune_logistic_regression(X_train, y_train, param_distributions, n_iter1):
    """
    Perform hyperparameter tuning using RandomizedSearchCV on a LogisticRegression model.
    
    Parameters:
    - X_train (pd.DataFrame or sparse matrix): Training features.
    - y_train (pd.Series): Training target.
    - param_distributions (dict): Hyperparameter search space.
    - n_iter1 (int): Number of parameter settings sampled.

    Returns:
    - best_model_LR (LogisticRegression): The best model found by RandomizedSearchCV.
    - best_params_LR (dict): The best parameters found by RandomizedSearchCV.
    """

    pos_weight = sum(y_train == 0) / sum(y_train == 1)
    class_weight = {0: 1, 1: pos_weight}  

    model = LogisticRegression(random_state=42, class_weight=class_weight)

    prauc_scorer = make_scorer(average_precision_score, needs_proba=True)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,  
        n_iter=n_iter1,
        scoring=prauc_scorer,
        cv=5,  
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    return random_search.best_estimator_, random_search.best_params_


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_layers, activation):
        super(NeuralNet, self).__init__()
        layers = []
        prev_size = input_size
        
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.01))
            prev_size = size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())  
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def compute_class_weight(y_train):
    y_train = y_train.astype(int)
    class_counts = np.bincount(y_train)
    weights = 1.0 / (class_counts + 1e-6)  
    weights = weights / weights.sum()  
    return torch.tensor(weights, dtype=torch.float32)

def train_model(X_train, y_train, hidden_layers, activation, alpha, learning_rate, max_iter, batch_size, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    class_weights = compute_class_weight(y_train).to(device)

    model = NeuralNet(X_train.shape[1], hidden_layers, activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=alpha)

    best_loss = float('inf')  
    best_model_state = None  
    early_stop_counter = 0  

    for epoch in range(max_iter):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()

            weights = class_weights[batch_y.long().view(-1)].view(-1, 1)  

            criterion = nn.BCELoss(weight=weights)  
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()  
            early_stop_counter = 0  
        else:
            early_stop_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        if early_stop_counter >= patience:
            print(f"Early Stopping at epoch {epoch}, Best Loss = {best_loss:.4f}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)  
        print(f"Loaded best model with Loss = {best_loss:.4f}")

    return model


def evaluate_model(model, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred_prob = model(X_tensor).cpu().numpy().flatten()  
    
    return average_precision_score(y_test, y_pred_prob)

def tune_neural_network(X_train, y_train, param_distributions, n_iter=10, k=5, patience=10):
    best_model = None
    best_params = None
    best_prauc = -np.inf

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for i in range(n_iter):
        params = {k: random.choice(v) for k, v in param_distributions.items()}
        print(f"Tuning iteration {i+1}/{n_iter} with params: {params}")

        fold_praucs = []  

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_train_split, X_val = X_train[train_idx], X_train[val_idx]
            y_train_split, y_val = y_train[train_idx], y_train[val_idx]

            model = train_model(
                X_train_split, y_train_split, 
                hidden_layers=params['hidden_layer_sizes'], 
                activation=params['activation'], 
                alpha=params['alpha'], 
                learning_rate=params['learning_rate'], 
                max_iter=params['max_iter'],
                batch_size=params['batch_size'],
                patience=patience  
            )

            prauc = evaluate_model(model, X_val, y_val)
            fold_praucs.append(prauc)

        mean_prauc = np.mean(fold_praucs)
        print(f"Average K-Fold PRAUC = {mean_prauc:.4f}")

        if mean_prauc > best_prauc:
            best_prauc = mean_prauc
            best_model = model
            best_params = params

    print(f"Best K-Fold PRAUC: {best_prauc:.4f} with params {best_params}")
    return best_model, best_params

def evaluate_model_with_checks_NN(test_data_dict, test_year, best_model, category, class_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)  
    best_model.eval()  

    test_data = test_data_dict[test_year]
    test_y = test_data['FLAG'].values  
    test_X = test_data.drop(
        columns=['ENCOUNTERID', 'SINCE_ADMIT', 'PATID', 'ADMIT_DATE', 'BCCOVID', 'FLAG'],
        errors='ignore'
    ).astype(np.float32)  

    if len(np.unique(test_y)) <= 1:
        return {
            'AUC': np.nan,
            'PRAUC': np.nan,
            'F1-Score': np.nan,
            'Precision': np.nan,
            'Recall': np.nan
        }

    try:
        test_X_tensor = torch.tensor(test_X.values, dtype=torch.float32).to(device)

        with torch.no_grad():
            test_pred_prob_tensor = best_model(test_X_tensor).cpu()  
        
        test_pred_prob = test_pred_prob_tensor.numpy().flatten()

        test_pred = (test_pred_prob >= 0.5).astype(int)

        auc = roc_auc_score(test_y, test_pred_prob)
        prauc = average_precision_score(test_y, test_pred_prob)
        f1 = f1_score(test_y, test_pred)
        precision_value = precision_score(test_y, test_pred)
        recall_value = recall_score(test_y, test_pred)

        return {
            'AUC': auc,
            'PRAUC': prauc,
            'F1-Score': f1,
            'Precision': precision_value,
            'Recall': recall_value
        }
    
    except Exception as e:
        return {
            'AUC': np.nan,
            'PRAUC': np.nan,
            'F1-Score': np.nan,
            'Precision': np.nan,
            'Recall': np.nan
        }

def evaluate_model_performance_with_cv_NN(data_by_year_test, test_year, best_model_NN):
    """
    Evaluate a PyTorch neural network model on test data.
    
    Parameters:
    - data_by_year_test (dict): Dictionary containing test data.
    - test_year (int): The year to test.
    - best_model_NN (torch.nn.Module): Trained PyTorch model.

    Returns:
    - dict: Evaluation metrics (AUC, PRAUC, F1, Precision, Recall).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_NN.to(device)
    best_model_NN.eval()  

    test_data = data_by_year_test[test_year]
    test_y = test_data['FLAG'].to_numpy().astype(np.float32)  
    test_X = test_data.drop(columns=['ENCOUNTERID', 'SINCE_ADMIT', 'PATID', 'ADMIT_DATE', 'BCCOVID', 'FLAG'], errors='ignore')

    test_X = test_X.astype(np.float32).to_numpy()
    test_X_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)

    with torch.no_grad():
        test_pred_prob_tensor = best_model_NN(test_X_tensor)
    
    test_pred_prob = test_pred_prob_tensor.cpu().numpy().flatten()

    test_pred = (test_pred_prob >= 0.5).astype(int)  

    auc = roc_auc_score(test_y, test_pred_prob)
    prauc = average_precision_score(test_y, test_pred_prob)
    f1 = f1_score(test_y, test_pred)
    precision_value = precision_score(test_y, test_pred)
    recall_value = recall_score(test_y, test_pred)

    return {
        'AUC': auc,
        'PRAUC': prauc,
        'F1': f1,
        'Precision': precision_value,
        'Recall': recall_value
    }
