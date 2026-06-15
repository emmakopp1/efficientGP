"""
SHAP analysis for FCS prediction — Nigeria and Chad
Neural network training + SHAP feature importance
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import shap
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

path_nigeria_shap = BASE_DIR / "data" / "new" / "nigeria" / "nigeria_shap.csv"
path_chad_shap    = BASE_DIR / "data" / "new" / "chad" / "chad-SHAP.csv"
dir_figures       = BASE_DIR / "output" / "figure"
dir_figures.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_nigeria(path):
    df = pd.read_csv(path, index_col=0)

    df = df.dropna(axis=1)
    df = df.drop(columns=["adm1_name", "cv_ind", "data_type", "day_of_week",
                           "rcsi", "Christian", "Lon", "Lat", "Easting", "Northing"])

    le = LabelEncoder()
    df["region_cat_encoded"] = le.fit_transform(df["region_cat"])
    print("Nigeria region encoding:")
    for i, cat in enumerate(le.classes_):
        print(f"  {cat} -> {i}")
    df = df.drop(columns=["region_cat"])

    df["Datetime"] = pd.to_datetime(df["Datetime"]).astype("int64") // 10**9

    features = df.drop(columns=["fcs"])
    target   = df["fcs"]

    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index,
    )
    return features_scaled, target


def load_chad(path):
    df = pd.read_csv(path, index_col=0)

    df = df.drop(columns=[
        "adm1_name", "adm1_code", "cv_ind", "day_of_week",
        "rcsi_cubic_intp", "fcs_cubic_intp", "pewi_cubic_intp",
        "Violence against civilians_cubic_intp", "Battles_cubic_intp",
        "Explosions/Remote violence_cubic_intp", "year_week",
        "rainfall_1_month_anomaly_cubic_intp", "ndvi_anomaly_cubic_intp",
        "rainfall_value_cubic_intp", "currency_exchange_cubic_intp",
        "Area", "Lon", "Lat", "Muslim", "Ramadan", "Ramadan_past90days",
    ])

    df["Datetime"] = pd.to_datetime(df["Datetime"]).astype("int64") // 10**9

    features = df.drop(columns=["fcs"])
    target   = df["fcs"]

    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index,
    )
    return features_scaled, target


# ============================================================================
# MODEL
# ============================================================================

class FCSDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features.values)
        self.targets  = torch.FloatTensor(targets.values).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class FCSPredictor(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32),          nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)


def train_model(model, train_loader, val_loader, num_epochs=200, lr=0.0001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                val_loss += criterion(model(X), y).item()

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}]  "
                  f"train={train_losses[-1]:.6f}  val={val_losses[-1]:.6f}")

    return train_losses, val_losses


def evaluate_model(model, loader):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.extend(model(X).cpu().numpy().flatten())
            actuals.extend(y.cpu().numpy().flatten())
    mse = mean_squared_error(actuals, preds)
    mae = mean_absolute_error(actuals, preds)
    return mse, mae, preds, actuals


# ============================================================================
# SHAP ANALYSIS
# ============================================================================

def run_shap_analysis(country, features_scaled, target):
    print(f"\n{'='*60}")
    print(f"  {country.upper()} — SHAP Analysis")
    print(f"{'='*60}")

    X_train, X_val, y_train, y_val = train_test_split(
        features_scaled, target, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

    train_loader = DataLoader(FCSDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader   = DataLoader(FCSDataset(X_val, y_val),   batch_size=32, shuffle=False)

    n_features = features_scaled.shape[1]
    model = FCSPredictor(n_features)
    print(f"Model: {n_features} input features\n")

    print("Training...")
    train_model(model, train_loader, val_loader)

    mse, mae, preds, actuals = evaluate_model(model, val_loader)
    print(f"\nValidation  MSE={mse:.6f}  MAE={mae:.6f}  RMSE={np.sqrt(mse):.6f}")

    print("\nSample predictions (first 10):")
    print(f"  Actual:    {[f'{a:.4f}' for a in actuals[:10]]}")
    print(f"  Predicted: {[f'{p:.4f}' for p in preds[:10]]}")

    # --- SHAP ---
    model.eval()

    def predict_fn(X):
        with torch.no_grad():
            return model(torch.FloatTensor(X)).numpy().flatten()

    print("\nBuilding SHAP explainer...")
    explainer  = shap.KernelExplainer(predict_fn, X_train)
    print("Computing SHAP values (this may take a few minutes)...")
    shap_values = explainer.shap_values(X_val)
    print("Done.")

    feature_importance = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"Variable": features_scaled.columns, "SHAP Importance": feature_importance})
        .sort_values("SHAP Importance", ascending=False)
    )

    print(f"\n{'='*50}")
    print("VARIABLE RANKING BY IMPORTANCE")
    print(f"{'='*50}")
    print(importance_df.to_string(index=False))

    # Mean SHAP Importance bar chart — saved to file
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(importance_df["Variable"][:15], importance_df["SHAP Importance"][:15])
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(f"{country} — Top 15 Mean SHAP Importance", fontsize=14)
    ax.invert_yaxis()
    fig.tight_layout()
    out_path = dir_figures / f"{country.lower()}_mean_shap_importance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Loading Nigeria data...")
    features_nga, target_nga = load_nigeria(path_nigeria_shap)

    print("Loading Chad data...")
    features_tcd, target_tcd = load_chad(path_chad_shap)

    run_shap_analysis("Nigeria", features_nga, target_nga)
    run_shap_analysis("Chad",    features_tcd, target_tcd)
