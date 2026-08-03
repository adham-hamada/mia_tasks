import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold

# 1. Load Data
X = pd.read_csv("data/X_train.csv").values
Y = pd.read_csv("data/y_train.csv").values.flatten()
zero_mask = pd.read_csv("data/test_zero_mask.csv")["zero_minutes"].values.astype(bool)

X_test = pd.read_csv("data/X_test.csv").values
test_ids = pd.read_csv("data/test_ids.csv")["Id"].values

# 2. Match PyTorch 5-Fold Split Exactly
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_pred = np.zeros(len(X))
test_pred_folds = np.zeros((5, len(X_test)))
fold_val_rmses = []

# 3. XGBoost Hyperparameters (Tuned for tabular regression & regularization)
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 6,          # Deep enough for positional crosses, shallow enough to avoid overfitting
    "subsample": 0.8,        # Train on 80% of rows per tree
    "colsample_bytree": 0.8, # Train on 80% of features per tree (forces exploration of non-dominant stats)
    "alpha": 0.1,            # L1 Regularization 
    "lambda": 1.0,           # L2 Regularization
    "random_state": 42
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/5")
    X_train_fold, Y_train_fold = X[train_idx], Y[train_idx]
    X_val_fold, Y_val_fold = X[val_idx], Y[val_idx]
    
    # XGBoost Data Matrices
    dtrain = xgb.DMatrix(X_train_fold, label=Y_train_fold)
    dval = xgb.DMatrix(X_val_fold, label=Y_val_fold)
    dtest = xgb.DMatrix(X_test)
    
    # Train with early stopping
    evals = [(dtrain, "train"), (dval, "val")]
    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=1500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Predict
    fold_val_pred = model.predict(dval)
    fold_test_pred = model.predict(dtest)
    
    oof_pred[val_idx] = fold_val_pred
    test_pred_folds[fold] = fold_test_pred
    
    # Calculate Fold RMSE
    fold_rmse = np.sqrt(((Y_val_fold - fold_val_pred) ** 2).mean())
    fold_val_rmses.append(fold_rmse)
    print(f"Fold {fold+1} val RMSE: {fold_rmse:.4f}")

# 4. Overall Cross-Validation Estimate
cv_rmse = np.sqrt(((Y - oof_pred) ** 2).mean())
print(f"\nMean of per-fold RMSEs: {np.mean(fold_val_rmses):.4f}  (+/- {np.std(fold_val_rmses):.4f})")
print(f"Overall out-of-fold RMSE (XGBoost): {cv_rmse:.4f}")

# 5. Process Final Predictions
test_pred_xgb = test_pred_folds.mean(axis=0)
test_pred_xgb[zero_mask] = 0.0
test_pred_xgb = np.clip(test_pred_xgb, 0.0, 10.0)

# Save standalone XGBoost submission
sub_xgb = pd.DataFrame({"Id": test_ids, "Player Rating": test_pred_xgb.round(2)})
sub_xgb.to_csv("submission_xgb.csv", index=False)
print(f"\nWrote standalone XGBoost predictions to submission_xgb.csv")

# 6. Automatic Blending with PyTorch
print("\n--- Generating Final Blend ---")
try:
    sub_nn = pd.read_csv("submission_v3.csv")
    # Simple 50/50 weighted average
    blended_rating = (sub_nn["Player Rating"] * 0.5) + (sub_xgb["Player Rating"] * 0.5)
    
    sub_blend = pd.DataFrame({"Id": test_ids, "Player Rating": blended_rating.round(2)})
    sub_blend.to_csv("submission_blend_final.csv", index=False)
    print("SUCCESS: Read submission_v3.csv and created 50/50 blend -> submission_blend_final.csv")
except FileNotFoundError:
    print("WARNING: submission_v3.csv not found. Run your PyTorch script first to enable blending.")