import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = pd.read_csv("data/X_train.csv").values.astype("float32")
Y = pd.read_csv("data/y_train.csv").values.astype("float32")
zero_mask = pd.read_csv("data/test_zero_mask.csv")["zero_minutes"].values.astype(bool)

X_test = pd.read_csv("data/X_test.csv").values.astype("float32")
test_ids = pd.read_csv("data/test_ids.csv")["Id"].values
# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)



class TabularDataset(Dataset):
    def __init__(self, X, y, w=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.w = torch.tensor(w, dtype=torch.float32) if w is not None else None
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, i):
        if self.w is not None:
            return self.X[i], self.y[i], self.w[i]
        return self.X[i], self.y[i]
    
# train_dataset = DataLoader(TabularDataset(X_train, Y_train), batch_size=64, shuffle=True)
# val_dataset = DataLoader(TabularDataset(X_val, Y_val), batch_size=256, shuffle=False)

torch.manual_seed(42)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def reduceLRonPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr
    )
    return scheduler

class MyModel(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 16) # Compress down to 16 latent features
        )
        
        # The "Wide" path combines the 16 deep features with the original raw inputs
        self.final = torch.nn.Linear(16 + input_dim, 1)

    def forward(self, x):
        deep_features = self.deep(x)
        combined = torch.cat([deep_features, x], dim=1)
        return self.final(combined)

def train_one_fold(X_train, Y_train, X_val, Y_val):
    train_dataset = DataLoader(TabularDataset(X_train, Y_train), batch_size=64, shuffle=True)
    val_dataset = DataLoader(TabularDataset(X_val, Y_val), batch_size=256, shuffle=False)
    model = MyModel(X_train.shape[1]).to(device)
    model.apply(init_weights)

    loss_fn = torch.nn.MSELoss(reduction="none")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = reduceLRonPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)

    early_stopping_patience = 30
    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(300):
        model.train()
        for X_batch, y_batch in train_dataset:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

        model.eval()
        val_loss = []
        with torch.no_grad():
            for X_batch, y_batch in val_dataset:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss.append(loss_fn(model(X_batch), y_batch).mean().item())
        
        epoch_val_loss = np.mean(val_loss)
        scheduler.step(epoch_val_loss)
        if epoch_val_loss < best_val_loss - 1e-4:
            best_val_loss = epoch_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1:3d}"
                f"val_mse={np.mean(val_loss):.4f}")

    model.load_state_dict(best_state)
    return model, best_val_loss

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_pred = np.zeros(len(X), dtype="float32")
test_pred_folds = np.zeros((5, len(X_test)), dtype="float32")

fold_val_rmses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/5")
    X_train_fold, Y_train_fold = X[train_idx], Y[train_idx]
    X_val_fold, Y_val_fold = X[val_idx], Y[val_idx]

    feat_mean = X_train_fold.mean(axis=0, keepdims=True)
    feat_std = X_train_fold.std(axis=0, keepdims=True)
    X_train = (X_train_fold - feat_mean) / (feat_std + 1e-8)
    X_val = (X_val_fold - feat_mean) / (feat_std + 1e-8)
    X_test_scaled = (X_test - feat_mean) / (feat_std + 1e-8)


    model, val_loss = train_one_fold(X_train, Y_train_fold, X_val, Y_val_fold)
    
    model.eval()
    with torch.no_grad():
        fold_val_pred = model(torch.tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy().flatten()
        fold_test_pred = model(torch.tensor(X_test_scaled, dtype=torch.float32, device=device)).cpu().numpy().flatten()
 
    oof_pred[val_idx] = fold_val_pred
    test_pred_folds[fold] = fold_test_pred
 
    fold_rmse = np.sqrt(((Y_val_fold.flatten() - fold_val_pred) ** 2).mean())
    fold_val_rmses.append(fold_rmse)
    print(f"Fold {fold+1} val RMSE: {fold_rmse:.4f}")
 
# ---- overall CV estimate (unbiased -- every row was validated on exactly once) ----
cv_rmse = np.sqrt(((Y.flatten() - oof_pred) ** 2).mean())
print(f"\nPer-fold RMSEs: {[round(r, 4) for r in fold_val_rmses]}")
print(f"Mean of per-fold RMSEs: {np.mean(fold_val_rmses):.4f}  (+/- {np.std(fold_val_rmses):.4f})")
print(f"Overall out-of-fold RMSE (the real CV estimate): {cv_rmse:.4f}")
 
test_pred = test_pred_folds.mean(axis=0)
test_pred[zero_mask] = 0.0
test_pred = np.clip(test_pred, 0.0, 10.0)

submission = pd.DataFrame({"Id": test_ids, "Player Rating": test_pred.round(2)})
submission.to_csv(f"submission_v3.csv", index=False)
print(f"\nWrote {len(submission)} rows to submission_v3.csv")