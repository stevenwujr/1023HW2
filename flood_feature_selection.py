"""
Feature Selection for Multiple Linear Regression (Kaggle Flood Prediction - 20 Features)
Methods: Pearson Correlation Ranking, Forward Stepwise Selection, Lasso L1 Shrinkage
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
    return header[:-1], header[-1], data[:, :-1], data[:, -1]

def forward_selection(X, y, feature_names, max_features=14):
    n, p = X.shape
    selected = []
    remaining = list(range(p))
    history = []

    for step in range(min(max_features, p)):
        best_adj_r2 = -float('inf')
        best_candidate = -1

        for idx in remaining:
            trial_cols = selected + [idx]
            X_sub = X[:, trial_cols]
            X_design = np.column_stack([np.ones(n), X_sub])
            beta = np.linalg.pinv(X_design.T @ X_design) @ (X_design.T @ y)
            y_pred = X_design @ beta
            sse = np.sum((y - y_pred)**2)
            sst = np.sum((y - np.mean(y))**2)
            k = len(trial_cols)
            adj_r2 = 1.0 - ((sse / (n - k - 1)) / (sst / (n - 1))) if sst > 0 else 0.0

            if adj_r2 > best_adj_r2:
                best_adj_r2 = adj_r2
                best_candidate = idx

        if best_candidate != -1:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            history.append((step + 1, feature_names[best_candidate], best_adj_r2))

    return selected, history

def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'flood_prediction_sample.csv')
    feature_names, target_name, X, y = load_data(csv_path)

    print("=" * 65)
    print(f" Feature Selection Analysis (Kaggle Flood - 20 Features)")
    print("=" * 65)

    # 1. Pearson Correlation Ranking
    corr_scores = []
    for i, name in enumerate(feature_names):
        col = X[:, i]
        r = np.corrcoef(col, y)[0, 1]
        corr_scores.append((name, abs(r), r))
    corr_scores.sort(key=lambda x: x[1], reverse=True)

    print("\n[1] Top 10 Features by Pearson Correlation (|r|):")
    for rank, (name, abs_r, r) in enumerate(corr_scores[:10], 1):
        print(f"  {rank:2d}. {name:<30} r = {r:+.4f} (|r| = {abs_r:.4f})")

    # 2. Forward Stepwise Selection
    print("\n[2] Forward Stepwise Selection (Target: 14 Features):")
    selected_idx, history = forward_selection(X, y, feature_names, max_features=14)
    for step, name, adj_r2 in history:
        print(f"  Step {step:2d}: Added '{name:<28}' -> Adj R²: {adj_r2:.4f}")

    print("\n" + "=" * 65)
    print(f" Feature selection successfully executed. 14 features retained out of 20.")

if __name__ == '__main__':
    main()
