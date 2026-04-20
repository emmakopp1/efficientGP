"""
MLP Nested Cross-Validation Runner
Run nested CV for FCS prediction on different countries

Usage:
To run training on Chad with data saving:

python mlp.py \
    --country chad \
    --project-path /Users/kopp/Documents/efficientGP/output/gaussian_process/TCD_gp_with_features_ridge_centered_20251216_114735/ \
    --output-data /Users/kopp/Documents/efficientGP/output/data/tcd_mlp

To run on specific folds only:

python mlp.py \
    --country nigeria \
    --project-path /Users/kopp/Documents/efficientGP/output/gaussian_process/NGA_gp_with_features_normal_centered_20260217_155435/ \
    --output-data /Users/kopp/Documents/efficientGP/output/data/nga_mlp
"""

import argparse
import pandas as pd
import numpy as np
import torch
import os
import json
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from utils.mlp_utils import (
    FCSDataset,
    FCSPredictor,
    train_model,
    evaluate_model,
    mc_dropout_uncertainty
)


def get_cv_splits(df, cv_fold):
    """
    Create custom train/validation splits for nested CV

    Args:
        df: DataFrame with 'cv_ind' column
        cv_fold: CV fold number (0-5)

    Returns:
        train_fold: Dict of training DataFrames for each config
        val_fold: Dict of validation DataFrames for each config
    """
    # Split by cv_ind
    cv = {i: df[df['cv_ind'] == i] for i in range(6)}

    train_fold = {}
    val_fold = {}

    # Define splits based on cv_fold
    if cv_fold == 0:
        train_fold[0] = pd.concat([cv[1], cv[2], cv[3], cv[4]])
        val_fold[0] = cv[5]
        train_fold[1] = pd.concat([cv[1], cv[2], cv[3], cv[5]])
        val_fold[1] = cv[4]
        train_fold[2] = pd.concat([cv[1], cv[2], cv[4], cv[5]])
        val_fold[2] = cv[3]
        train_fold[3] = pd.concat([cv[2], cv[3], cv[4], cv[5]])
        val_fold[3] = cv[1]
        train_fold[4] = pd.concat([cv[1], cv[3], cv[4], cv[5]])
        val_fold[4] = cv[2]

    elif cv_fold == 1:
        train_fold[0] = pd.concat([cv[0], cv[2], cv[3], cv[5]])
        val_fold[0] = cv[4]
        train_fold[1] = pd.concat([cv[0], cv[3], cv[4], cv[5]])
        val_fold[1] = cv[2]
        train_fold[2] = pd.concat([cv[0], cv[2], cv[4], cv[5]])
        val_fold[2] = cv[3]
        train_fold[3] = pd.concat([cv[0], cv[2], cv[3], cv[4]])
        val_fold[3] = cv[5]
        train_fold[4] = pd.concat([cv[2], cv[3], cv[4], cv[5]])
        val_fold[4] = cv[0]

    elif cv_fold == 2:
        train_fold[0] = pd.concat([cv[0], cv[1], cv[3], cv[5]])
        val_fold[0] = cv[4]
        train_fold[1] = pd.concat([cv[0], cv[3], cv[4], cv[5]])
        val_fold[1] = cv[1]
        train_fold[2] = pd.concat([cv[0], cv[1], cv[4], cv[5]])
        val_fold[2] = cv[3]
        train_fold[3] = pd.concat([cv[0], cv[1], cv[3], cv[4]])
        val_fold[3] = cv[5]
        train_fold[4] = pd.concat([cv[1], cv[3], cv[4], cv[5]])
        val_fold[4] = cv[0]

    elif cv_fold == 3:
        train_fold[0] = pd.concat([cv[0], cv[1], cv[2], cv[5]])
        val_fold[0] = cv[4]
        train_fold[1] = pd.concat([cv[0], cv[2], cv[4], cv[5]])
        val_fold[1] = cv[1]
        train_fold[2] = pd.concat([cv[0], cv[1], cv[4], cv[5]])
        val_fold[2] = cv[2]
        train_fold[3] = pd.concat([cv[0], cv[1], cv[2], cv[4]])
        val_fold[3] = cv[5]
        train_fold[4] = pd.concat([cv[1], cv[2], cv[4], cv[5]])
        val_fold[4] = cv[0]

    elif cv_fold == 4:
        train_fold[0] = pd.concat([cv[0], cv[1], cv[3], cv[5]])
        val_fold[0] = cv[2]
        train_fold[1] = pd.concat([cv[0], cv[3], cv[2], cv[5]])
        val_fold[1] = cv[1]
        train_fold[2] = pd.concat([cv[0], cv[1], cv[2], cv[5]])
        val_fold[2] = cv[3]
        train_fold[3] = pd.concat([cv[0], cv[1], cv[3], cv[2]])
        val_fold[3] = cv[5]
        train_fold[4] = pd.concat([cv[1], cv[2], cv[3], cv[5]])
        val_fold[4] = cv[0]

    elif cv_fold == 5:
        train_fold[0] = pd.concat([cv[0], cv[1], cv[2], cv[3]])
        val_fold[0] = cv[4]
        train_fold[1] = pd.concat([cv[0], cv[1], cv[2], cv[4]])
        val_fold[1] = cv[3]
        train_fold[2] = pd.concat([cv[0], cv[1], cv[3], cv[4]])
        val_fold[2] = cv[2]
        train_fold[3] = pd.concat([cv[1], cv[2], cv[3], cv[4]])
        val_fold[3] = cv[0]
        train_fold[4] = pd.concat([cv[0], cv[1], cv[2], cv[3]])
        val_fold[4] = cv[4]

    return train_fold, val_fold


def run_nested_cv_fold(train_fold, val_fold, num_epochs=200, lr=0.0001, verbose=True):
    """
    Run nested cross-validation for one outer fold

    Returns:
        best_model, best_scaler, val_mse_list, best_fold_idx
    """
    val_mse_list = []
    best_val_mse_list = []
    models = []
    scalers = []

    for i in range(5):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Config {i+1}/5")
            print(f"{'='*60}")

        train_df = train_fold[i]
        val_df = val_fold[i]

        # Prepare features
        X_train = train_df.drop(['adm1_name', 'cv_ind', 'fcs'], axis=1)
        X_train['Datetime'] = pd.to_datetime(X_train['Datetime']).astype('int64') // 10**9
        y_train = train_df['fcs']

        X_val = val_df.drop(['adm1_name', 'cv_ind', 'fcs'], axis=1)
        X_val['Datetime'] = pd.to_datetime(X_val['Datetime']).astype('int64') // 10**9
        y_val = val_df['fcs']

        # Normalization
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )

        # Create datasets
        train_dataset = FCSDataset(X_train_scaled, y_train)
        val_dataset = FCSDataset(X_val_scaled, y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Train model
        n_features = X_train_scaled.shape[1]
        model = FCSPredictor(n_features)

        model, train_losses, val_losses, best_val_mse = train_model(
            model, train_loader, val_loader,
            num_epochs=num_epochs, lr=lr, verbose=verbose, plot=False
        )

        # Evaluate
        mse, _, _, _ = evaluate_model(model, val_loader)

        val_mse_list.append(mse)
        best_val_mse_list.append(best_val_mse)
        models.append(model)
        scalers.append(scaler)

        if verbose:
            print(f"Config {i+1}: Best Val MSE = {best_val_mse:.6f}, Final Val MSE = {mse:.6f}")

    # Select best model
    best_fold_idx = np.argmin(val_mse_list)
    best_model = models[best_fold_idx]
    best_scaler = scalers[best_fold_idx]

    if verbose:
        print(f"\n{'='*60}")
        print(f"NESTED CV SUMMARY")
        print(f"{'='*60}")
        for i in range(5):
            print(f"Config {i+1}: Validation MSE = {val_mse_list[i]:.6f}")
        print(f"\nBest config: {best_fold_idx+1} (Val MSE = {val_mse_list[best_fold_idx]:.6f})")
        print(f"Mean CV MSE: {np.mean(val_mse_list):.6f} ± {np.std(val_mse_list):.6f}")

    return best_model, best_scaler, val_mse_list, best_fold_idx


def evaluate_test_set(test_df, best_model, best_scaler, cv_fold, output_data_path=None, n_mc_samples=100):
    """
    Evaluate on test set and generate plots

    Args:
        output_data_path: Optional path to save detailed predictions CSV
    """
    # Prepare features
    test_features = test_df.drop(['adm1_name', 'cv_ind', 'fcs'], axis=1)
    test_features['Datetime'] = pd.to_datetime(test_features['Datetime']).astype('int64') // 10**9
    test_features_scaled = pd.DataFrame(
        best_scaler.transform(test_features),
        columns=test_features.columns,
        index=test_features.index
    )
    test_targets = test_df['fcs']

    test_dataset = FCSDataset(test_features_scaled, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Get deterministic predictions
    test_mse, _, test_preds_eval, test_actuals = evaluate_model(best_model, test_loader)
    print(f"\nTest MSE (eval mode) = {test_mse:.6f}")

    # Get uncertainty estimates via MC Dropout
    _, std_preds_mc = mc_dropout_uncertainty(best_model, test_loader, n_samples=n_mc_samples)

    # Calculate confidence intervals
    test_preds_eval = np.array(test_preds_eval)
    lower_ci_mc = test_preds_eval - 1.96 * std_preds_mc
    upper_ci_mc = test_preds_eval + 1.96 * std_preds_mc

    # Build DataFrame for plotting
    df_plot = pd.DataFrame({
        'Datetime': pd.to_datetime(test_df['Datetime']),
        'y_hat': test_preds_eval,
        'y_true': test_df['fcs'],
        'ci_low': lower_ci_mc,
        'ci_high': upper_ci_mc,
        'adm1_name': test_df['adm1_name']
    })

    # Compute RMSE by region
    rmse_by_region = df_plot.groupby('adm1_name', group_keys=False).apply(
        lambda g: np.sqrt(np.mean((g['y_hat'] - g['y_true'])**2))
    )
    df_plot = df_plot.merge(rmse_by_region.rename('rMSE'), left_on='adm1_name', right_index=True)

    df_plot['y_residual'] = df_plot['y_hat'] - df_plot['y_true']

    # Save detailed predictions to CSV if output_data_path is provided
    if output_data_path is not None:
        os.makedirs(output_data_path, exist_ok=True)
        predictions_file = os.path.join(output_data_path, f'predictions_cv{cv_fold}.csv')
        df_plot.to_csv(predictions_file, index=False)
        print(f"Predictions saved to {predictions_file}")

    return df_plot, test_mse


def main():
    parser = argparse.ArgumentParser(description='Run MLP Nested Cross-Validation')
    parser.add_argument('--country', type=str, required=True,
                        help='Country name (e.g., chad, nigeria)')
    parser.add_argument('--project-path', type=str, required=True,
                        help='Path to project directory containing data_train_cvX.csv files')
    parser.add_argument('--output-data', type=str, default=None,
                        help='Path to save detailed predictions CSV files (default: None)')
    parser.add_argument('--num-epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--cv-folds', type=str, default='0,1,2,3,4,5',
                        help='Comma-separated CV fold indices to run (default: 0,1,2,3,4,5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Parse CV folds
    cv_folds = [int(x) for x in args.cv_folds.split(',')]

    print(f"Running MLP Nested CV for {args.country.upper()}")
    print(f"Project path: {args.project_path}")
    print(f"Output data: {args.output_data if args.output_data else 'Not saving'}")
    print(f"CV Folds: {cv_folds}")
    print(f"Epochs: {args.num_epochs}, LR: {args.lr}")

    all_results = []
    all_predictions = []

    for cv_fold in cv_folds:
        print(f"\n{'#'*80}")
        print(f"# OUTER CV FOLD {cv_fold}")
        print(f"{'#'*80}")

        # Load training data
        train_path = f"{args.project_path}/data_train_cv{cv_fold}.csv"
        df = pd.read_csv(train_path)
        print(f"Loaded training data: {df.shape}")

        # Get CV splits
        train_fold, val_fold = get_cv_splits(df, cv_fold)

        # Run nested CV
        best_model, best_scaler, val_mse_list, best_fold_idx = run_nested_cv_fold(
            train_fold, val_fold,
            num_epochs=args.num_epochs,
            lr=args.lr,
            verbose=True
        )

        # Evaluate on test set
        test_path = f"{args.project_path}/data_test_cv{cv_fold}.csv"
        test_df = pd.read_csv(test_path)
        print(f"\nLoaded test data: {test_df.shape}")

        df_plot, test_mse = evaluate_test_set(
            test_df, best_model, best_scaler,
            cv_fold=cv_fold,
            output_data_path=args.output_data
        )

        # Add cv_fold column to df_plot for combined dataset
        df_plot['cv_fold'] = cv_fold
        all_predictions.append(df_plot)

        all_results.append({
            'cv_fold': cv_fold,
            'best_config': best_fold_idx,
            'val_mse_list': val_mse_list,
            'test_mse': test_mse
        })

    # Summary
    print(f"\n{'#'*80}")
    print(f"# FINAL SUMMARY - {args.country.upper()}")
    print(f"{'#'*80}")
    for result in all_results:
        print(f"CV Fold {result['cv_fold']}: Test MSE = {result['test_mse']:.6f}, "
              f"Best Config = {result['best_config']}")

    mean_test_mse = np.mean([r['test_mse'] for r in all_results])
    std_test_mse = np.std([r['test_mse'] for r in all_results])
    print(f"\nOverall Test MSE: {mean_test_mse:.6f} ± {std_test_mse:.6f}")

    # Save combined predictions and summary if output_data is specified
    if args.output_data is not None:
        os.makedirs(args.output_data, exist_ok=True)

        # Save combined predictions from all folds
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        combined_file = os.path.join(args.output_data, f'{args.country}_all_predictions.csv')
        combined_predictions.to_csv(combined_file, index=False)
        print(f"\nCombined predictions saved to {combined_file}")

        # Save summary results
        summary = {
            'country': args.country,
            'mean_test_mse': float(mean_test_mse),
            'std_test_mse': float(std_test_mse),
            'cv_folds': []
        }

        for result in all_results:
            summary['cv_folds'].append({
                'cv_fold': int(result['cv_fold']),
                'best_config': int(result['best_config']),
                'test_mse': float(result['test_mse']),
                'val_mse_list': [float(x) for x in result['val_mse_list']]
            })

        summary_file = os.path.join(args.output_data, f'{args.country}_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")

        # Calculate and print overall metrics
        print(f"\n{'='*80}")
        print(f"SAVED DATA SUMMARY")
        print(f"{'='*80}")
        print(f"Total predictions: {len(combined_predictions)}")
        print(f"Number of regions: {combined_predictions['adm1_name'].nunique()}")
        print(f"Date range: {combined_predictions['Datetime'].min()} to {combined_predictions['Datetime'].max()}")

        # Calculate overall confidence interval coverage
        coverage = ((combined_predictions['y_true'] >= combined_predictions['ci_low']) &
                   (combined_predictions['y_true'] <= combined_predictions['ci_high'])).mean()
        print(f"Overall CI coverage: {coverage*100:.2f}%")

        # Calculate overall RMSE
        overall_rmse = np.sqrt(np.mean((combined_predictions['y_hat'] - combined_predictions['y_true'])**2))
        print(f"Overall RMSE: {overall_rmse:.6f}")


if __name__ == '__main__':
    main()
