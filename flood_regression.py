"""
Multiple Linear Regression on Kaggle Flood Prediction (20 Features)
Dataset: Kaggle Playground Series s4e5
Student ID: 7115056112 | Data Mining HW2
"""

import os
import sys
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def load_data(csv_path: str):
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    with open(csv_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
    feature_names = header[:-1]
    target_name = header[-1]
    X = data[:, :-1]
    y = data[:, -1]
    print(f"Dataset: {len(X)} samples, {len(feature_names)} features, Target: '{target_name}'")
    return feature_names, target_name, X, y

def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    n = len(X)
    indices = np.random.permutation(n)
    test_count = int(n * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def evaluate(y_true, y_pred, p):
    n = len(y_true)
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    sst = np.sum((y_true - np.mean(y_true))**2)
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    r2 = 1.0 - (sse / sst) if sst > 0 else 0.0
    adj_r2 = 1.0 - ((sse / (n - p - 1)) / (sst / (n - 1))) if sst > 0 and n > p + 1 else r2
    return {"rmse": rmse, "mae": mae, "r2": r2, "adj_r2": adj_r2}

def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'flood_prediction_sample.csv')
    feature_names, target_name, X, y = load_data(csv_path)

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    n_train, p = X_train.shape
    print(f"Training on {n_train} samples with {p} features...")

    # Design matrix with intercept
    X_tr_design = np.column_stack([np.ones(n_train), X_train])
    beta = np.linalg.pinv(X_tr_design.T @ X_tr_design) @ (X_tr_design.T @ y_train)

    # Train evaluation
    y_pred_tr = X_tr_design @ beta
    tr_metrics = evaluate(y_train, y_pred_tr, p)

    # Test evaluation
    X_te_design = np.column_stack([np.ones(len(X_test)), X_test])
    y_pred_te = X_te_design @ beta
    te_metrics = evaluate(y_test, y_pred_te, p)

    print("\n" + "=" * 50)
    print(f" Train R²:  {tr_metrics['r2']:.4f} | Adj R²: {tr_metrics['adj_r2']:.4f}")
    print(f" Test R²:   {te_metrics['r2']:.4f} | Adj R²: {te_metrics['adj_r2']:.4f}")
    print(f" Test RMSE: {te_metrics['rmse']:.4f} | Test MAE: {te_metrics['mae']:.4f}")
    print("=" * 50)

if __name__ == '__main__':
    main()
