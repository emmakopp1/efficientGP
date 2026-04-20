"""
MLP Functions and Classes for FCS Prediction
Extracted from MLP_Nested_CV_chad.ipynb
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.dates as mdates
import copy


# ============================================================================
# Dataset and Model Classes
# ============================================================================

class FCSDataset(Dataset):
    """PyTorch Dataset for FCS data"""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features.values)
        self.targets = torch.FloatTensor(targets.values).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class FCSPredictor(nn.Module):
    """Neural Network Model for FCS Prediction"""
    def __init__(self, n_features):
        super(FCSPredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, verbose=True, plot=True):
    """
    Train the model with early stopping based on validation MSE

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs to train
        lr: Learning rate
        verbose: If True, print training progress
        plot: If True, display training curves

    Returns:
        model: Trained model with best weights loaded
        train_losses: List of training losses per epoch
        val_losses: List of validation MSEs per epoch
        best_val_mse: Best validation MSE achieved
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_val_mse = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_mse, _, _, _ = evaluate_model(model, val_loader)
        val_losses.append(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val MSE: {val_mse:.6f}')

    model.load_state_dict(best_model_state)

    if verbose:
        print(f'\nBest validation MSE: {best_val_mse:.6f}')

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', linewidth=2)
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation MSE', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Courbes de Loss pendant l\'entrainement')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return model, train_losses, val_losses, best_val_mse


def evaluate_model(model, test_loader):
    """
    Evaluate the model on test/validation data

    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test/validation data

    Returns:
        mse: Mean Squared Error
        mae: Mean Absolute Error
        predictions: List of predicted values
        actuals: List of actual values
    """
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for features, targets in test_loader:
            outputs = model(features)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(targets.cpu().numpy().flatten())

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)

    return mse, mae, predictions, actuals


# ============================================================================
# MC Dropout for Uncertainty Estimation
# ============================================================================

def mc_dropout_uncertainty(model, test_loader, n_samples=100):
    """
    Perform MC Dropout to estimate prediction uncertainty

    Args:
        model: PyTorch model with dropout layers
        test_loader: DataLoader for test data
        n_samples: Number of forward passes to perform

    Returns:
        mean_preds: Mean predictions across all samples
        std_preds: Standard deviation of predictions
    """
    model.train()  # Enable dropout during inference
    all_preds = []

    with torch.no_grad():  # No gradient computation needed
        for _ in range(n_samples):  # Repeat n_samples times
            preds = []
            for features, _ in test_loader:
                outputs = model(features)  # Make predictions with dropout active
                preds.extend(outputs.cpu().numpy().flatten())
            all_preds.append(preds)

    all_preds = np.array(all_preds)  # Shape: (n_samples, num_test_points)
    return np.mean(all_preds, axis=0), np.std(all_preds, axis=0)


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_time_series_predictions(df, cv_fold, output_path):
    """
    Reproduces time series plots with error bars for each point

    Args:
        df: DataFrame with columns: Datetime, y_hat, y_true, ci_low, ci_high, adm1_name, rMSE
        cv_fold: Cross-validation fold number
        output_path: Path to save the plot
    """
    # Get unique regions
    regions = df['adm1_name'].unique()

    # Define grid layout
    n_cols = 2
    n_rows = int(np.ceil(len(regions) / n_cols))

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    # Color palette
    colors = plt.cm.Set1(np.linspace(0, 1, len(regions)))

    for idx, region in enumerate(regions):
        ax = axes[idx]

        # Filter data for this region
        region_data = df[df['adm1_name'] == region].sort_values('Datetime')
        rmse = region_data['rMSE'].iloc[0] # first rmse value for the region

        # Calculate coverage: percentage of true values within confidence interval
        within_ci = ((region_data['y_true'] >= region_data['ci_low']) &
                     (region_data['y_true'] <= region_data['ci_high']))
        coverage = (within_ci.sum() / len(region_data)) * 100

        # Calculate average length of 95% confidence interval
        avg_ci_length = (region_data['ci_high'] - region_data['ci_low']).mean()

        # Convert Datetime to matplotlib dates if needed
        if not pd.api.types.is_datetime64_any_dtype(region_data['Datetime']):
            region_data['Datetime'] = pd.to_datetime(region_data['Datetime'])


        # Plot error bars for confidence intervals
        ax.errorbar(region_data['Datetime'].values,
            region_data['y_hat'].values,
            yerr=[
                (region_data['y_hat'] - region_data['ci_low']).values,
                (region_data['ci_high'] - region_data['y_hat']).values
            ],
            #fmt='none',
            ecolor='lightgray',
            alpha=0.7,
            capsize=2,
            capthick=0.5,
            elinewidth=0.8,
            label='Confidence interval' if idx == 0 else ''
            )

        # Plot predictions (dashed line)
        ax.plot(region_data['Datetime'].values,
                region_data['y_hat'].values,
                linestyle='--',
                color=colors[idx],
                linewidth=1,
                label='Predicted values' if idx == 0 else '',
                marker='.',
                markersize=3)

        # Plot true values (black points)
        ax.scatter(region_data['Datetime'].values,
                   region_data['y_true'].values,
                   color='black',
                   s=15,
                   zorder=5,
                   label='True' if idx == 0 else '')

        # Title with RMSE, coverage, and average CI length
        ax.set_title(f"{region} (rMSE: {rmse:.3f}, Coverage: {coverage:.1f}%, Avg CI: {avg_ci_length:.3f})")

        # Set y-axis limits between 0 and 1
        ax.set_ylim(0, 1)

        # Format x-axis to show dates properly
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Rotate date labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)

        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle=':')

        # Legend only for first subplot
        if idx == 0:
            ax.legend(loc='best', fontsize=8)

    # Remove empty axes
    for idx in range(len(regions), len(axes)):
        fig.delaxes(axes[idx])

    # General title
    fig.suptitle(f'Test set - CV Fold {cv_fold}', fontsize=14, y=1.02)

    # Adjust layout
    plt.tight_layout()

    #print(output_path)

    # Save
    plt.savefig(f"{output_path}/time_series_prediction_cv{cv_fold}.png",
                dpi=100, bbox_inches='tight')
    plt.close()


def plot_residuals_distribution(df, cv_fold, output_path):
    """
    Reproduces residuals distribution histogram by region

    Args:
        df: DataFrame with columns: y_residual, adm1_name
        cv_fold: Cross-validation fold number
        output_path: Path to save the plot
    """
    regions = df['adm1_name'].unique()
    n_regions = len(regions)

    n_cols = int(np.ceil(np.sqrt(n_regions)))
    n_rows = int(np.ceil(n_regions / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, n_regions))

    # Calculate global x-axis limits to align zero across all plots
    all_residuals = df['y_residual']
    x_min = all_residuals.min()
    x_max = all_residuals.max()

    # Make limits symmetric around zero for better alignment
    x_limit = max(abs(x_min), abs(x_max))
    x_range = (-x_limit, x_limit)

    for idx, region in enumerate(regions):
        ax = axes[idx]

        # Filter data for this region
        region_data = df[df['adm1_name'] == region]

        # Create histogram
        ax.hist(region_data['y_residual'],
                bins=40,
                alpha=0.7,
                color=colors[idx],
                range=x_range)

        # Set x-axis limits to align zero
        ax.set_xlim(x_range)

        # Add vertical line at zero (black dashed)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)

        # Title with region name only
        ax.set_title(region)

        # Labels
        ax.set_xlabel('Residuals (Predicted - True)', fontsize=8)
        ax.set_ylabel('Count')

        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    # Remove empty axes
    for idx in range(n_regions, len(axes)):
        fig.delaxes(axes[idx])

    # General title
    fig.suptitle(f'Distribution of Residuals by Region',
                 fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(f"{output_path}/residuals_distribution_cv{cv_fold}.png",
                dpi=100, bbox_inches='tight')
    plt.close()
