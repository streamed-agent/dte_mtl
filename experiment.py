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

def seed(i=123):
  torch.manual_seed(i)
  random.seed(i)
  np.random.seed(i)

COLOR_PALETTE = ["royalblue", "red", "skyblue", "orange", "green", "purple", "brown", "pink", "gray", "olive", "cyan"]
method_label = {"single": "Single Task (NN)", "linear": "Single Task (Linear)", "multi": "Multi Task (NN)", "monotonic": "Multi Task (Monotonic NN)", "empirical": "Empirical"}
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

def run_single_simulation(
    n: int,
    locations: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Run a single simulation iteration."""
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
        LinearRegression(), is_multi_task=False, folds=2
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
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    wrapper = TorchModelWrapper(model=model, criterion=criterion, optimizer=optimizer, epochs=100, batch_size=16)
    nn_estimator = dte_adj.AdjustedDistributionEstimator(
        wrapper, is_multi_task=False, folds=2
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
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    wrapper = TorchModelWrapper(model=model, criterion=criterion, optimizer=optimizer, epochs=100, batch_size=16)
    nn_estimator = dte_adj.AdjustedDistributionEstimator(
        wrapper, is_multi_task=True, folds=2
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
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    wrapper = TorchModelWrapper(model=model, criterion=criterion, optimizer=optimizer, epochs=100, batch_size=16)
    nn_estimator = dte_adj.AdjustedDistributionEstimator(
        wrapper, is_multi_task=True, folds=2
    )
    nn_estimator.fit(X, D, Y)
    results['monotonic'] = nn_estimator.predict_dte(
        TREATMENT_ARM, CONTROL_ARM, locations, variance_type="moment"
    )
    execution_times['monotonic'] = time.time() - start_time

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

    # Print summary statistics
    print("\nExecution time summary (seconds):")
    for method in results.keys():
        times = execution_times[method]
        print(f"{method:>10}: mean={np.mean(times):.4f}, std={np.std(times):.4f}")

    # Create visualizations
    print("Creating plots...")
    create_performance_plots(df, locations, n, list(results.keys()))

    print("Simulation completed successfully!")

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
    run_simulation(500, key="_500")
    run_water_consumption()

if __name__ == "__main__":
    main()
    
