import dte_adj
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
import torch.nn.functional as F
import pandas as pd
import random
from collections import defaultdict
import tqdm
from sklearn.metrics import accuracy_score
from dte_mtl_baselines import MMoENN, CrossStitchNN, CascadedNN
from scipy import stats
import pickle

def seed(i=123):
  torch.manual_seed(i)
  random.seed(i)
  np.random.seed(i)

COLOR_PALETTE = ["royalblue", "red", "skyblue", "orange", "green", "purple", "brown", "pink", "gray", "olive", "cyan"]
method_label = {
    "single": "Single Task (NN)",
    "linear": "Single Task (Linear)",
    "multi": "Multi Task (NN)",
    "monotonic": "Multi Task (Monotonic NN)",
    "empirical": "Empirical",
    "mmoe": "MMoE",
    "cross_stitch": "Cross-Stitch",
    "cascaded": "Cascaded"
}
FONT_SIZE = 18

def generate_data(n, d_x=20, rho=0.5):
    """
    Generate data according to the described data generating process (DGP).
    Args:
    n (int): Number of samples.
    d_x (int): Number of covariates. Default is 20.
    rho (float): Success probability for the Bernoulli distribution. Default is 0.5.
    Returns:
    X (np.ndarray): Covariates matrix of shape (n, d_x).
    D (np.ndarray): Treatment variable array of shape (n,).
    Y (np.ndarray): Outcome variable array of shape (n,).
    """
    # Generate covariates X from a uniform distribution on (0, 1)
    X = np.random.uniform(0, 1, (n, d_x))
    
    # Generate treatment variable D from a Bernoulli distribution with success probability rho
    D = np.random.binomial(1, rho, n)
    
    # Define beta_j and gamma_j according to the problem statement
    beta = np.zeros(d_x)
    gamma = np.zeros(d_x)
    
    # Set the first 50 values of beta and gamma to 1
    beta[:18] = 1
    gamma[:10] = 1
    
    # Compute the outcome Y
    U = np.random.normal(0, 1, n)  # Error term
    
    # Outcome equation
    Y = np.where(D[:, np.newaxis] , np.ones(d_x) * X, beta * X)

    Y = Y[:, :, np.newaxis] * Y[:, np.newaxis, :]
    Y = Y.reshape(n, -1).sum(axis=-1) + U
    
    return X, D, Y

class TorchModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, criterion, optimizer, epochs=10, batch_size=32, device=None, patience=50, verbose=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.best_loss = float('inf')
        self.counter = 0
        self.patience = patience
        self.verbose = verbose

    def fit(self, X, y):
        # Convert inputs to PyTorch tensors and move to the selected device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.reshape(-1, 1)
        
        self.model.train()

        for epoch in range(self.epochs):
            permutation = torch.randperm(X_tensor.size()[0])
            total_loss = 0
            for i in range(0, X_tensor.size()[0], self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_x, batch_y = X_tensor[indices], y_tensor[indices]

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                total_loss += loss.item()
                self.optimizer.step()
            if self.verbose:
                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {total_loss}')

        return self

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return ((outputs>0.5) * 1).cpu().numpy()

    def predict_proba(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            # probabilities = torch.softmax(outputs, dim=1)
        probabilities = outputs.cpu().squeeze().numpy()
        return probabilities

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        if(x.dim() == 1):
            x = x.reshape(-1, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.sigmoid(x)

class CumulativeNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CumulativeNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        if(x.dim() == 1):
            x = x.reshape(-1, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.clamp(x, max=10)
        # x = torch.cumsum(F.relu(x), dim=1)
        x = torch.cumsum(torch.exp(x), dim=1)
        # x = F.sigmoid(x) * 2 - 1
        x = torch.atan(x) / (torch.pi / 2)
        
        return x

def calculate_metrics(
    results: dict[str, list],
    true_dte: np.ndarray,
    locations: np.ndarray,
    execution_times: np.ndarray,
) -> pd.DataFrame:
    """Calculate performance metrics from simulation results."""
    metrics_data = {"locations": locations}
    methods = list(results.keys())

    for method in methods:
        method_results = np.array(results[method])
        times = np.array(execution_times[method])
        point_estimates = method_results[:, 0]
        lower_bounds = method_results[:, 1]
        upper_bounds = method_results[:, 2]

        # Calculate metrics
        interval_lengths = (upper_bounds - lower_bounds).mean(axis=0)
        coverage_prob = ((upper_bounds >= true_dte) & (true_dte >= lower_bounds)).mean(axis=0)
        rmse = np.sqrt(((point_estimates - true_dte) ** 2).mean(axis=0))
        bias = ((point_estimates.mean() - true_dte)**2).mean(axis=0)
        time_avg = times.mean(axis=0)
        time_std = times.std(axis=0)

        metrics_data.update({
            f"interval length - {method}": interval_lengths,
            f"coverage probability - {method}": coverage_prob,
            f"RMSE - {method}": rmse,
            f"bias - {method}": bias,
            f"execution time avg - {method}": time_avg,
            f"execution time std - {method}": time_std,
        })

    df = pd.DataFrame(metrics_data)

    # Calculate RMSE reductions
    for method in methods:
        df[f"RMSE reduction (%) {method} / empirical"] = (
            1 - df[f"RMSE - {method}"] / df["RMSE - empirical"]
        ) * 100
    return df

def diebold_mariano_test(errors1, errors2, h=1):
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Tests H0: errors1 and errors2 have equal predictive accuracy
    vs H1: errors1 has worse accuracy than errors2

    Args:
        errors1: Forecast errors from method 1 (baseline) - shape (n_simulations, n_locations)
        errors2: Forecast errors from method 2 (comparison) - shape (n_simulations, n_locations)
        h: Forecast horizon (default=1 for one-step ahead)

    Returns:
        dm_stat: DM test statistic
        p_value: One-sided p-value (tests if method 2 is better)
    """
    # Compute loss differential (squared errors)
    d = errors1**2 - errors2**2

    # Mean loss differential
    d_mean = d.mean()

    # Variance of loss differential with HAC correction
    n = len(d.flatten())
    d_flat = d.flatten()
    d_var = np.var(d_flat, ddof=1) / n

    # DM statistic
    dm_stat = d_mean / np.sqrt(d_var)

    # P-value (one-sided: testing if method 2 is better)
    p_value = 1 - stats.norm.cdf(dm_stat)

    return dm_stat, p_value

def compute_statistical_tests(
    results: dict,
    true_dte: np.ndarray,
    baseline_method: str = 'empirical'
) -> pd.DataFrame:
    """
    Perform statistical significance tests comparing all methods against baseline.

    Args:
        results: Dictionary with method names as keys and list of (estimate, lower, upper) tuples
        true_dte: Ground truth DTE values
        baseline_method: Name of baseline method to compare against

    Returns:
        DataFrame with test results
    """
    test_results = []
    methods = [m for m in results.keys() if m != baseline_method]

    # Extract point estimates for all methods
    baseline_estimates = np.array([r[0] for r in results[baseline_method]])
    n_simulations = len(baseline_estimates)

    # Compute errors for baseline
    baseline_errors = baseline_estimates - true_dte

    for method in methods:
        method_estimates = np.array([r[0] for r in results[method]])
        method_errors = method_estimates - true_dte

        # Compute RMSE for each simulation (averaged across quantiles)
        baseline_rmse_per_sim = np.sqrt((baseline_errors**2).mean(axis=1))
        method_rmse_per_sim = np.sqrt((method_errors**2).mean(axis=1))

        # Compute average RMSE reduction
        rmse_reductions = (baseline_rmse_per_sim - method_rmse_per_sim) / baseline_rmse_per_sim * 100
        mean_reduction = rmse_reductions.mean()
        std_reduction = rmse_reductions.std()

        # Paired t-test (H1: baseline RMSE > method RMSE, i.e., method is better)
        t_stat, p_value_t = stats.ttest_rel(
            baseline_rmse_per_sim,
            method_rmse_per_sim,
            alternative='greater'
        )

        # Wilcoxon signed-rank test (non-parametric alternative)
        try:
            w_stat, p_value_w = stats.wilcoxon(
                baseline_rmse_per_sim,
                method_rmse_per_sim,
                alternative='greater'
            )
        except:
            w_stat, p_value_w = np.nan, np.nan

        # Diebold-Mariano test (for forecast comparison)
        dm_stat, p_value_dm = diebold_mariano_test(baseline_errors, method_errors)

        # Determine significance level
        if p_value_t < 0.001:
            sig_star = '***'
        elif p_value_t < 0.01:
            sig_star = '**'
        elif p_value_t < 0.05:
            sig_star = '*'
        else:
            sig_star = ''

        test_results.append({
            'Method': method_label.get(method, method),
            'Mean RMSE Reduction (%)': f"{mean_reduction:.2f}",
            'Std RMSE Reduction (%)': f"{std_reduction:.2f}",
            't-statistic': f"{t_stat:.3f}",
            'p-value (t-test)': f"{p_value_t:.4f}" if p_value_t >= 0.0001 else "<0.0001",
            'p-value (Wilcoxon)': f"{p_value_w:.4f}" if not np.isnan(p_value_w) and p_value_w >= 0.0001 else "<0.0001",
            'DM statistic': f"{dm_stat:.3f}",
            'p-value (DM)': f"{p_value_dm:.4f}" if p_value_dm >= 0.0001 else "<0.0001",
            'Significance': sig_star,
            'Sample Size': n_simulations
        })

    return pd.DataFrame(test_results)

def add_significance_stars_to_summary(
    results: dict,
    true_dte: np.ndarray,
    baseline_method: str = 'empirical'
) -> dict:
    """
    Compute summary statistics with significance stars for each method.

    Returns dictionary with summary stats including significance information.
    """
    summary_with_stars = {}
    baseline_estimates = np.array([r[0] for r in results[baseline_method]])
    baseline_errors = baseline_estimates - true_dte

    for method in results.keys():
        if method == baseline_method:
            continue

        method_estimates = np.array([r[0] for r in results[method]])
        method_errors = method_estimates - true_dte

        # Compute RMSE for each quantile across simulations
        rmse_per_quantile = np.sqrt((method_errors**2).mean(axis=0))
        baseline_rmse_per_quantile = np.sqrt((baseline_errors**2).mean(axis=0))

        # Compute reduction for each quantile
        reduction_per_quantile = (baseline_rmse_per_quantile - rmse_per_quantile) / baseline_rmse_per_quantile * 100

        # Test significance for each quantile
        quantile_p_values = []
        for q_idx in range(method_errors.shape[1]):
            baseline_rmse_per_sim = np.sqrt((baseline_errors[:, q_idx]**2))
            method_rmse_per_sim = np.sqrt((method_errors[:, q_idx]**2))
            _, p_val = stats.ttest_rel(baseline_rmse_per_sim, method_rmse_per_sim, alternative='greater')
            quantile_p_values.append(p_val)

        quantile_p_values = np.array(quantile_p_values)

        # Summary statistics
        summary_with_stars[method] = {
            'min': reduction_per_quantile.min(),
            'p25': np.percentile(reduction_per_quantile, 25),
            'p50': np.percentile(reduction_per_quantile, 50),
            'p75': np.percentile(reduction_per_quantile, 75),
            'max': reduction_per_quantile.max(),
            'p_values': quantile_p_values,
            'all_significant': np.all(quantile_p_values < 0.05),
            'most_significant': np.mean(quantile_p_values < 0.001) > 0.8
        }

    return summary_with_stars

def generate_latex_significance_table(test_df: pd.DataFrame, output_file: str = None) -> str:
    """
    Generate LaTeX table from statistical test results.

    Args:
        test_df: DataFrame with test results from compute_statistical_tests
        output_file: Optional file path to save LaTeX code

    Returns:
        LaTeX table code as string
    """
    latex_lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Statistical Significance Tests for RMSE Reduction vs. Empirical Estimator}",
        "\\label{tab:statistical_tests}",
        "\\small",
        "\\begin{tabular}{|l|c|c|c|c|}",
        "\\hline",
        "\\textbf{Method} & \\textbf{Mean Reduction (\\%)} & \\textbf{t-statistic} & \\textbf{p-value} & \\textbf{Sig.} \\\\",
        "\\hline"
    ]

    for _, row in test_df.iterrows():
        line = f"{row['Method']} & {row['Mean RMSE Reduction (%)']} & {row['t-statistic']} & {row['p-value (t-test)']} & {row['Significance']} \\\\"
        latex_lines.append(line)

    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\begin{minipage}{0.9\\textwidth}",
        "\\textit{Notes:} Paired t-tests compare RMSE (averaged across quantiles) between each method and the empirical estimator. ",
        f"N={test_df['Sample Size'].iloc[0]} simulations. Significance levels: *** p<0.001, ** p<0.01, * p<0.05.",
        "\\end{minipage}",
        "\\end{table}"
    ])

    latex_code = "\n".join(latex_lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex_code)
        print(f"LaTeX table saved to {output_file}")

    return latex_code

def create_performance_plots(df: pd.DataFrame, locations: np.ndarray, n: int, methods: list[str]) -> None:
    """Create visualization plots for simulation results.
    
    Args:
        df: DataFrame containing the performance metrics
        locations: Array of location values for x-axis
        n: Sample size for title
        methods: List of method names (e.g., ["empirical", "linear", "xgb"])
                Methods should match the suffixes in DataFrame column names
    """
    # Create color mapping based on method order
    method_colors = {method: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, method in enumerate(methods)}
    
    # Define metric patterns to look for in DataFrame columns
    metric_patterns = {
        'RMSE': 'RMSE - {}',
        'Average CI Length': 'interval length - {}', 
        'Coverage Probability': 'coverage probability - {}'
    }
    
    # Create 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(15, 7), sharex=True)
    axs = axs.flatten()  # Flatten for easier indexing
    
    # Performance metrics comparison (first 3 subplots)
    for i, (title, pattern) in enumerate(metric_patterns.items()):
        if title == "RMSE":
            ax = axs[0]
        else:
            ax = axs[i+1]
        plotted_methods = []
        
        for method in methods:
            column_name = pattern.format(method)
            if column_name in df.columns:
                ax.plot(locations, df[column_name], 
                       label=method_label.get(method, method), marker='o', color=method_colors[method])
                plotted_methods.append(method)
        
        if 'Coverage Probability' in title:
            ax.set_ylim(0.9, 1)
        ax.set_title(title, fontsize=FONT_SIZE)
        if title != "RMSE":
            ax.set_xlabel("Quantiles", fontsize=FONT_SIZE)
        # if i == 0:
        #     ax.set_ylabel("Value", fontsize=FONT_SIZE)
        ax.grid(True)
    
    # RMSE reduction plot (4th subplot)
    ax = axs[1]
    reduction_columns = [col for col in df.columns if 'RMSE reduction' in col and '/' in col]
    plotted_any = False
    
    for method in methods:
        if method == 'empirical':
            continue
        # Look for reduction columns involving this method
        method_reduction_cols = [col for col in reduction_columns if method in col.lower()]
        
        for col in method_reduction_cols:
            ax.plot(
                locations, df[col],
                color=method_colors[method], marker='o', 
                label=method_label.get(method, method)
            )
            plotted_any = True
    
    if plotted_any:
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_title("RMSE Reduction", fontsize=FONT_SIZE)
        # ax.set_xlabel("Quantiles", fontsize=FONT_SIZE)
        # ax.set_ylabel("RMSE Reduction (%)", fontsize=FONT_SIZE)
        ax.grid(True)
    
    # Create a single legend for all subplots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(methods), fontsize=FONT_SIZE-2, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust to make room for legend
    plt.show()

TREATMENT_ARM=1
CONTROL_ARM=0

# Default hyperparameters
DEFAULT_HYPERPARAMS = {
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 0.01,
    'hidden_dims': [128, 64],  # [h1, h2]
    'folds': 2
}

def run_single_simulation(
    n: int,
    locations: np.ndarray,
    hyperparams: dict = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Run a single simulation iteration with specified hyperparameters."""
    if hyperparams is None:
        hyperparams = DEFAULT_HYPERPARAMS.copy()

    data = generate_data(n=n)
    X, D, Y = data

    results = {}
    execution_times = {}

    # Empirical estimator
    start_time = time.time()
    empirical_estimator = dte_adj.SimpleDistributionEstimator()
    empirical_estimator.fit(X, D, Y)
    results['empirical'] = empirical_estimator.predict_dte(
        TREATMENT_ARM, CONTROL_ARM, locations, variance_type="moment"
    )
    execution_times['empirical'] = time.time() - start_time

    # Linear adjusted estimator
    start_time = time.time()
    linear_estimator = dte_adj.AdjustedDistributionEstimator(
        LinearRegression(), is_multi_task=False, folds=hyperparams['folds']
    )
    linear_estimator.fit(X, D, Y)
    results['linear'] = linear_estimator.predict_dte(
        TREATMENT_ARM, CONTROL_ARM, locations, variance_type="moment"
    )
    execution_times['linear'] = time.time() - start_time

    # single
    start_time = time.time()
    model = SimpleNN(input_dim=X.shape[1], output_dim=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    wrapper = TorchModelWrapper(model=model, criterion=criterion, optimizer=optimizer,
                                  epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'])
    nn_estimator = dte_adj.AdjustedDistributionEstimator(
        wrapper, is_multi_task=False, folds=hyperparams['folds']
    )
    nn_estimator.fit(X, D, Y)
    results['single'] = nn_estimator.predict_dte(
        TREATMENT_ARM, CONTROL_ARM, locations, variance_type="moment"
    )
    execution_times['single'] = time.time() - start_time

    # multi
    start_time = time.time()
    model = SimpleNN(input_dim=X.shape[1], output_dim=len(locations))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    wrapper = TorchModelWrapper(model=model, criterion=criterion, optimizer=optimizer,
                                  epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'])
    nn_estimator = dte_adj.AdjustedDistributionEstimator(
        wrapper, is_multi_task=True, folds=hyperparams['folds']
    )
    nn_estimator.fit(X, D, Y)
    results['multi'] = nn_estimator.predict_dte(
        TREATMENT_ARM, CONTROL_ARM, locations, variance_type="moment"
    )
    execution_times['multi'] = time.time() - start_time

    # Monotonic
    start_time = time.time()
    model = CumulativeNN(input_dim=X.shape[1], output_dim=len(locations))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    wrapper = TorchModelWrapper(model=model, criterion=criterion, optimizer=optimizer,
                                  epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'])
    nn_estimator = dte_adj.AdjustedDistributionEstimator(
        wrapper, is_multi_task=True, folds=hyperparams['folds']
    )
    nn_estimator.fit(X, D, Y)
    results['monotonic'] = nn_estimator.predict_dte(
        TREATMENT_ARM, CONTROL_ARM, locations, variance_type="moment"
    )
    execution_times['monotonic'] = time.time() - start_time

    # MMoE (Tier 1 baseline)
    start_time = time.time()
    model = MMoENN(input_dim=X.shape[1], output_dim=len(locations), num_experts=3, expert_hidden=64)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    wrapper = TorchModelWrapper(
        model=model, criterion=criterion, optimizer=optimizer,
        epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size']
    )
    nn_estimator = dte_adj.AdjustedDistributionEstimator(
        wrapper, is_multi_task=True, folds=hyperparams['folds']
    )
    nn_estimator.fit(X, D, Y)
    results['mmoe'] = nn_estimator.predict_dte(
        TREATMENT_ARM, CONTROL_ARM, locations, variance_type="moment"
    )
    execution_times['mmoe'] = time.time() - start_time

    # Cross-Stitch (Tier 2 baseline)
    start_time = time.time()
    model = CrossStitchNN(input_dim=X.shape[1], output_dim=len(locations))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    wrapper = TorchModelWrapper(
        model=model, criterion=criterion, optimizer=optimizer,
        epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size']
    )
    nn_estimator = dte_adj.AdjustedDistributionEstimator(
        wrapper, is_multi_task=True, folds=hyperparams['folds']
    )
    nn_estimator.fit(X, D, Y)
    results['cross_stitch'] = nn_estimator.predict_dte(
        TREATMENT_ARM, CONTROL_ARM, locations, variance_type="moment"
    )
    execution_times['cross_stitch'] = time.time() - start_time

    # Cascaded (Tier 2 baseline)
    start_time = time.time()
    model = CascadedNN(input_dim=X.shape[1], output_dim=len(locations))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    wrapper = TorchModelWrapper(
        model=model, criterion=criterion, optimizer=optimizer,
        epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size']
    )
    nn_estimator = dte_adj.AdjustedDistributionEstimator(
        wrapper, is_multi_task=True, folds=hyperparams['folds']
    )
    nn_estimator.fit(X, D, Y)
    results['cascaded'] = nn_estimator.predict_dte(
        TREATMENT_ARM, CONTROL_ARM, locations, variance_type="moment"
    )
    execution_times['cascaded'] = time.time() - start_time

    return results, execution_times

results = defaultdict(list)
execution_times = defaultdict(list)
def run_simulation(iterations=100, n=1000, key: str=""):
    seed(123)
    global results
    global execution_times

    # compute test statistics
    X_test, D_test, Y_test = generate_data(10**5)
    locations = np.array([np.quantile(Y_test, i*0.05) for i in range(1,20)])
    estimator = dte_adj.SimpleDistributionEstimator()
    estimator.fit(X_test, D_test, Y_test)
    dte_test, _, _ = estimator.predict_dte(target_treatment_arm=TREATMENT_ARM, control_treatment_arm=CONTROL_ARM, locations=locations, variance_type="simple")

    for epoch in tqdm.tqdm(range(iterations)):
        iter_results, iter_times = run_single_simulation(
            n, locations
        )

        for method in iter_results.keys():
            results[method].append(iter_results[method])
            execution_times[method].append(iter_times[method])

    # Calculate and save metrics
    print("Calculating metrics...")
    df = calculate_metrics(results, dte_test, locations, execution_times)
    output_file = f"dte_{n}{key}.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Save raw results for reproducibility and re-analysis
    raw_results = {
        'results': dict(results),  # Convert defaultdict to dict
        'execution_times': dict(execution_times),
        'true_dte': dte_test,
        'locations': locations,
        'n': n,
        'iterations': iterations,
        'seed': 123
    }
    raw_output_file = f"dte_{n}{key}_raw.pkl"
    with open(raw_output_file, 'wb') as f:
        pickle.dump(raw_results, f)
    print(f"Raw results saved to {raw_output_file}")

    # Perform statistical significance tests
    print("\nPerforming statistical significance tests...")
    test_df = compute_statistical_tests(results, dte_test, baseline_method='empirical')

    # Save test results
    test_output_file = f"statistical_tests_{n}{key}.csv"
    test_df.to_csv(test_output_file, index=False)
    print(f"Statistical test results saved to {test_output_file}")

    # Generate and save LaTeX table
    latex_output_file = f"statistical_tests_{n}{key}.tex"
    generate_latex_significance_table(test_df, output_file=latex_output_file)

    # Compute summary statistics with significance
    summary_stats = add_significance_stars_to_summary(results, dte_test, baseline_method='empirical')

    # Print test results
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TEST RESULTS")
    print("="*80)
    print(test_df.to_string(index=False))
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")
    print("="*80)

    # Print summary statistics with significance
    print("\n" + "="*80)
    print("SUMMARY STATISTICS WITH SIGNIFICANCE")
    print("="*80)
    for method, stats_dict in summary_stats.items():
        sig_str = "***" if stats_dict['most_significant'] else ("*" if stats_dict['all_significant'] else "")
        print(f"\n{method_label.get(method, method)}:")
        print(f"  min={stats_dict['min']:.1f}, p25={stats_dict['p25']:.1f}, p50={stats_dict['p50']:.1f}, "
              f"p75={stats_dict['p75']:.1f}, max={stats_dict['max']:.1f} {sig_str}")
        print(f"  All quantiles significant (p<0.05): {stats_dict['all_significant']}")
        print(f"  Most quantiles highly significant (p<0.001): {stats_dict['most_significant']}")
    print("="*80)

    # Print summary statistics
    print("\nExecution time summary (seconds):")
    for method in results.keys():
        times = execution_times[method]
        print(f"{method:>10}: mean={np.mean(times):.4f}, std={np.std(times):.4f}")

    # Create visualizations
    print("\nCreating plots...")
    create_performance_plots(df, locations, n, list(results.keys()))

    print("\nSimulation completed successfully!")

def run_sensitivity_analysis(
    n=1000,
    iterations=50,
    param_grid=None,
    key="_sensitivity"
):
    """
    Run hyperparameter sensitivity analysis for multi-task DTE estimation.

    Tests different hyperparameter combinations to assess robustness and identify
    optimal settings. Results are saved for each hyperparameter configuration.

    Args:
        n: Sample size for each simulation
        iterations: Number of simulation runs per hyperparameter setting (default: 50)
        param_grid: Dictionary mapping hyperparameter names to lists of values to test.
                   Default grid tests learning rate, batch size, epochs, and folds.
        key: Suffix for output files

    Example:
        >>> # Test learning rates and batch sizes
        >>> param_grid = {
        ...     'learning_rate': [0.001, 0.01, 0.1],
        ...     'batch_size': [8, 16, 32]
        ... }
        >>> run_sensitivity_analysis(n=1000, iterations=50, param_grid=param_grid)
    """
    from itertools import product

    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [8, 16, 32],
            'epochs': [50, 100, 200],
            'folds': [2, 5]
        }

    # Generate all hyperparameter combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    param_combinations = list(product(*param_values))

    print("="*80)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"Sample size: {n}")
    print(f"Iterations per configuration: {iterations}")
    print(f"Number of configurations: {len(param_combinations)}")
    print(f"\nParameter grid:")
    for name, values in param_grid.items():
        print(f"  {name}: {values}")
    print("="*80)

    # Prepare ground truth
    seed(123)
    X_test, D_test, Y_test = generate_data(10**5)
    locations = np.array([np.quantile(Y_test, i*0.05) for i in range(1,20)])
    estimator = dte_adj.SimpleDistributionEstimator()
    estimator.fit(X_test, D_test, Y_test)
    dte_test, _, _ = estimator.predict_dte(
        target_treatment_arm=TREATMENT_ARM,
        control_treatment_arm=CONTROL_ARM,
        locations=locations,
        variance_type="simple"
    )

    # Store results for all configurations
    all_results = []

    # Test each hyperparameter combination
    for config_idx, param_combo in enumerate(param_combinations):
        # Create hyperparams dict for this configuration
        hyperparams = DEFAULT_HYPERPARAMS.copy()
        for param_name, param_value in zip(param_names, param_combo):
            hyperparams[param_name] = param_value

        config_str = "_".join([f"{name}={value}" for name, value in zip(param_names, param_combo)])
        print(f"\n[{config_idx+1}/{len(param_combinations)}] Testing configuration: {config_str}")

        # Run simulations with this hyperparameter configuration
        config_results = defaultdict(list)
        config_times = defaultdict(list)

        for epoch in tqdm.tqdm(range(iterations), desc=f"Config {config_idx+1}"):
            iter_results, iter_times = run_single_simulation(
                n, locations, hyperparams=hyperparams
            )

            for method in iter_results.keys():
                config_results[method].append(iter_results[method])
                config_times[method].append(iter_times[method])

        # Calculate metrics for this configuration
        df = calculate_metrics(config_results, dte_test, locations, config_times)

        # Add hyperparameter values to results
        for param_name, param_value in zip(param_names, param_combo):
            df[param_name] = param_value

        all_results.append(df)

        # Save individual configuration results
        config_file = f"sensitivity_{config_str}{key}.csv"
        df.to_csv(config_file, index=False)
        print(f"Results saved to {config_file}")

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_file = f"sensitivity_all{key}.csv"
    combined_df.to_csv(combined_file, index=False)
    print(f"\n{'='*80}")
    print(f"All sensitivity results saved to {combined_file}")

    # Generate sensitivity analysis summary
    print("\nGenerating sensitivity analysis summary...")
    sensitivity_summary = []

    # Extract method names from the results
    methods = list(config_results.keys())

    for method in methods:
        method_data = combined_df[combined_df['method'] == method]

        for param_name in param_names:
            # Group by parameter value and compute mean RMSE reduction
            grouped = method_data.groupby(param_name)['rmse_reduction_pct'].agg(['mean', 'std', 'min', 'max'])

            for param_value in param_grid[param_name]:
                if param_value in grouped.index:
                    stats = grouped.loc[param_value]
                    sensitivity_summary.append({
                        'method': method,
                        'parameter': param_name,
                        'value': param_value,
                        'mean_rmse_reduction': stats['mean'],
                        'std_rmse_reduction': stats['std'],
                        'min_rmse_reduction': stats['min'],
                        'max_rmse_reduction': stats['max']
                    })

    sensitivity_df = pd.DataFrame(sensitivity_summary)
    sensitivity_file = f"sensitivity_summary{key}.csv"
    sensitivity_df.to_csv(sensitivity_file, index=False)
    print(f"Sensitivity summary saved to {sensitivity_file}")

    # Print summary results
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*80)

    for method in methods:
        print(f"\n{method_label.get(method, method)}:")
        method_data = sensitivity_df[sensitivity_df['method'] == method]

        for param_name in param_names:
            param_data = method_data[method_data['parameter'] == param_name]
            if len(param_data) > 0:
                print(f"\n  {param_name}:")
                for _, row in param_data.iterrows():
                    print(f"    {row['value']}: {row['mean_rmse_reduction']:.1f}% "
                          f"(std={row['std_rmse_reduction']:.1f})")

    print("\n" + "="*80)
    print("Sensitivity analysis completed!")
    print("="*80)

    return combined_df, sensitivity_df


def load_raw_results(filename):
    """
    Load raw simulation results from pickle file.

    Args:
        filename: Path to the pickle file containing raw results

    Returns:
        Dictionary containing:
        - results: Dict of method -> list of (dte, var, ci) tuples
        - execution_times: Dict of method -> list of execution times
        - true_dte: Ground truth DTE values
        - locations: Quantile locations
        - n: Sample size
        - iterations: Number of simulation iterations
        - seed: Random seed used
    """
    with open(filename, 'rb') as f:
        raw_results = pickle.load(f)
    return raw_results


def rerun_statistical_tests(raw_results_file, baseline_method='empirical', output_prefix=None):
    """
    Rerun statistical significance tests from saved raw results.

    Args:
        raw_results_file: Path to pickle file with raw simulation results
        baseline_method: Which method to use as baseline (default: 'empirical')
        output_prefix: Prefix for output files. If None, uses input filename prefix

    Returns:
        test_df: DataFrame with statistical test results
    """
    # Load raw results
    print(f"Loading raw results from {raw_results_file}...")
    raw_results = load_raw_results(raw_results_file)

    results = raw_results['results']
    true_dte = raw_results['true_dte']
    locations = raw_results['locations']
    n = raw_results['n']
    iterations = raw_results['iterations']

    print(f"Loaded results: n={n}, iterations={iterations}, methods={list(results.keys())}")

    # Compute statistical tests
    print("\nPerforming statistical significance tests...")
    test_df = compute_statistical_tests(results, true_dte, baseline_method=baseline_method)

    # Determine output prefix
    if output_prefix is None:
        output_prefix = raw_results_file.replace('_raw.pkl', '')

    # Save test results
    test_output_file = f"{output_prefix}_rerun_tests.csv"
    test_df.to_csv(test_output_file, index=False)
    print(f"Statistical test results saved to {test_output_file}")

    # Generate and save LaTeX table
    latex_output_file = f"{output_prefix}_rerun_tests.tex"
    generate_latex_significance_table(test_df, output_file=latex_output_file)
    print(f"LaTeX table saved to {latex_output_file}")

    # Print results
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TEST RESULTS")
    print("="*80)
    print(test_df.to_string(index=False))
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")
    print("="*80)

    return test_df


def run_water_consumption():
    df_water = pd.read_csv("path/to//090113_TotWatDat_cor_merge_Price.tab", sep='\t')
    df_water = df_water.dropna()
    locations = np.arange(1, 200)
    X = df_water[[
        "jun06", "jul06", "aug06", "sep06", "oct06", "nov06", "dec06", 
        "jan07", "feb07", "mar07", "apr07_3x", "may07_3x"
    ]].values
    Y = (df_water.jun07_3x+df_water.jul07_3x+df_water.aug07_3x+df_water.sep07_3x).values
    D = df_water.treatment.values

    seed(123)
    estimator = dte_adj.SimpleDistributionEstimator()
    estimator.fit(X,D,Y)
    empirical_dte, empirical_lower_bound, empirical_upper_bound = estimator.predict_dte(target_treatment_arm=3, control_treatment_arm=4,
                                                                                        locations=locations, variance_type="multiplier", n_bootstrap=5000)

    seed(123)
    estimator = dte_adj.AdjustedDistributionEstimator(LinearRegression(), is_multi_task=False, folds=2)
    estimator.fit(X, D, Y)
    linear_dte, linear_lower_bound, linear_upper_bound = estimator.predict_dte(target_treatment_arm=3, control_treatment_arm=4,
                                                                               locations=locations, variance_type="multiplier", n_bootstrap=5000)

    seed(123)
    model = SimpleNN(input_dim=X.shape[1], output_dim=len(locations))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    wrapper = TorchModelWrapper(model=model, criterion=criterion, optimizer=optimizer, epochs=50, batch_size=64)
    estimator = dte_adj.AdjustedDistributionEstimator(wrapper, is_multi_task=True, folds=2)
    _X = (X - X.mean()) / (X.std() + 0.0001)
    estimator.fit(_X, D, Y)
    multi_dte, multi_lower_bound, multi_upper_bound = estimator.predict_dte(target_treatment_arm=3, control_treatment_arm=4, locations=locations, variance_type="multiplier", n_bootstrap=5000)

    seed(123)
    model = CumulativeNN(input_dim=X.shape[1], output_dim=len(locations))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    wrapper = TorchModelWrapper(model=model, criterion=criterion, optimizer=optimizer, epochs=50, batch_size=64)
    estimator = dte_adj.AdjustedDistributionEstimator(wrapper, is_multi_task=True, folds=2)
    _X = (X - X.mean()) / (X.std() + 0.0001)
    estimator.fit(_X, D, Y)
    monotonic_dte, monotonic_lower_bound, monotonic_upper_bound = estimator.predict_dte(target_treatment_arm=3, control_treatment_arm=4, locations=locations, variance_type="multiplier", n_bootstrap=5000)

    seed(123)
    model = SimpleNN(input_dim=X.shape[1], output_dim=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    wrapper = TorchModelWrapper(model=model, criterion=criterion, optimizer=optimizer, epochs=30, batch_size=64)
    estimator = dte_adj.AdjustedDistributionEstimator(wrapper, is_multi_task=False, folds=2)
    _X = (X - X.mean()) / (X.std() + 0.0001)
    estimator.fit(_X, D, Y)
    single_dte, single_lower_bound, single_upper_bound = estimator.predict_dte(target_treatment_arm=3, control_treatment_arm=4, locations=locations, variance_type="multiplier", n_bootstrap=5000)

    pd.DataFrame({
      'empirical_dte': empirical_dte,
      'empirical_lower_bound': empirical_lower_bound,
      'empirical_upper_bound': empirical_upper_bound,
      'single_dte': single_dte,
      'single_lower_bound': single_lower_bound,
      'single_upper_bound': single_upper_bound,
      'monotonic_dte': monotonic_dte,
      'monotonic_lower_bound': monotonic_lower_bound,
      'monotonic_upper_bound': monotonic_upper_bound,
      'multi_dte': multi_dte,
      'multi_lower_bound': multi_lower_bound,
      'multi_upper_bound': multi_upper_bound,
      'linear_dte': linear_dte,
      'linear_lower_bound': linear_lower_bound,
      'linear_upper_bound': linear_upper_bound,
    }, index=locations).to_csv("water_consumption.csv")

    plt.plot(locations, 100-(monotonic_dte-monotonic_lower_bound) / (empirical_dte-empirical_lower_bound)*100, label="Multi Task (Monotonic NN)", color="green")
    plt.plot(locations, 100-(multi_dte-multi_lower_bound) / (empirical_dte-empirical_lower_bound)*100, label="Multi Task (NN)", color="orange")
    plt.plot(locations, 100-(single_dte-single_lower_bound) / (empirical_dte-empirical_lower_bound)*100, label="Single Task (NN)", color="lightblue")
    plt.plot(locations, 100-(linear_dte-linear_lower_bound) / (empirical_dte-empirical_lower_bound)*100, label="Single Task (Linear)", color="red")
    plt.legend(fontsize=10)
    plt.tick_params(labelsize=12)
    plt.xlabel("Water Consumption", fontsize=15)
    plt.ylabel("SE reduction (%)", fontsize=15)
    plt.show()
    

def main():
    run_simulation(500, key="_stat")
    run_water_consumption()

if __name__ == "__main__":
    main()
