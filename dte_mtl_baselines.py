"""
Multi-Task Learning Baseline Architectures for DTE Estimation

This module implements three baseline multi-task architectures:
1. MMoE (Multi-gate Mixture-of-Experts) - Tier 1 baseline
2. Cross-Stitch Networks - Tier 2 baseline
3. Cascaded Architecture - Tier 2 baseline

All architectures are designed to have comparable parameter counts to the
original SimpleNN and CumulativeNN models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMoENN(nn.Module):
    """
    Multi-gate Mixture-of-Experts Neural Network

    Based on: "Modeling Task Relationships in Multi-task Learning with
    Multi-gate Mixture-of-Experts" (Ma et al., KDD 2018)

    Architecture:
    - Multiple expert networks share computation
    - Each task has its own gating network to weight experts
    - Parameter count kept comparable to SimpleNN

    Args:
        input_dim: Input feature dimension
        output_dim: Number of tasks/outputs (number of threshold locations)
        num_experts: Number of expert networks (default: 3)
        expert_hidden: Hidden dimension for each expert (default: 64)
    """
    def __init__(self, input_dim, output_dim, num_experts=3, expert_hidden=64):
        super(MMoENN, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = output_dim

        # Shared bottom layer - maps input to common representation
        self.shared_bottom = nn.Linear(input_dim, 128)

        # Expert networks - each processes the shared representation
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, expert_hidden),
                nn.ReLU()
            ) for _ in range(num_experts)
        ])

        # Gate networks - one per task, decides which experts to use
        self.gates = nn.ModuleList([
            nn.Linear(128, num_experts) for _ in range(output_dim)
        ])

        # Tower networks - final prediction layer per task
        self.towers = nn.ModuleList([
            nn.Linear(expert_hidden, 1) for _ in range(output_dim)
        ])

    def forward(self, x):
        if x.dim() == 1:
            x = x.reshape(-1, 1)

        # Shared bottom processing
        shared = F.relu(self.shared_bottom(x))

        # Process through all experts
        expert_outputs = torch.stack([expert(shared) for expert in self.experts], dim=1)
        # expert_outputs shape: (batch_size, num_experts, expert_hidden)

        # For each task, gate the experts and produce output
        task_outputs = []
        for task_idx in range(self.num_tasks):
            # Compute gate weights for this task
            gate_logits = self.gates[task_idx](shared)
            gate_weights = F.softmax(gate_logits, dim=1)  # (batch_size, num_experts)

            # Weighted combination of expert outputs
            gate_weights_expanded = gate_weights.unsqueeze(2)  # (batch_size, num_experts, 1)
            weighted_expert = (expert_outputs * gate_weights_expanded).sum(dim=1)  # (batch_size, expert_hidden)

            # Final task-specific prediction
            task_output = self.towers[task_idx](weighted_expert)
            task_outputs.append(task_output)

        # Concatenate all task outputs
        output = torch.cat(task_outputs, dim=1)  # (batch_size, num_tasks)
        return F.sigmoid(output)


class CrossStitchNN(nn.Module):
    """
    Cross-Stitch Network

    Based on: "Cross-stitch Networks for Multi-task Learning" (Misra et al., CVPR 2016)

    Architecture:
    - Maintains both shared and task-specific representations
    - Cross-stitch units learn linear combinations between shared/task-specific features
    - Uses single shared representation for all tasks (simplified for parameter efficiency)

    Args:
        input_dim: Input feature dimension
        output_dim: Number of tasks/outputs
        hidden_dims: List of hidden layer dimensions (default: [128, 64])
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64]):
        super(CrossStitchNN, self).__init__()
        self.num_tasks = output_dim
        self.num_layers = len(hidden_dims)

        # Shared network path
        self.shared_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Task-specific paths (lightweight - only final layer)
        # To keep parameters comparable, we use a single task-specific path
        # that branches at the last layer
        self.task_specific_layer = nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2)

        # Cross-stitch units - learn to combine shared and task-specific features
        # Using a simplified version: single cross-stitch before final layer
        self.cross_stitch = nn.Parameter(torch.eye(2))  # 2x2 for [shared, task-specific]

        # Final output layers per task
        combined_dim = hidden_dims[-1] + hidden_dims[-1] // 2
        self.output_layers = nn.ModuleList([
            nn.Linear(combined_dim, 1) for _ in range(output_dim)
        ])

    def forward(self, x):
        if x.dim() == 1:
            x = x.reshape(-1, 1)

        # Forward through shared layers
        shared = x
        for layer in self.shared_layers:
            shared = F.relu(layer(shared))

        # Task-specific processing
        task_specific = F.relu(self.task_specific_layer(shared))

        # Apply cross-stitch: linear combination of shared and task-specific
        # Simplified: just concatenate weighted features
        alpha = torch.sigmoid(self.cross_stitch[0, 0])
        beta = torch.sigmoid(self.cross_stitch[1, 1])

        combined = torch.cat([alpha * shared, beta * task_specific], dim=1)

        # Generate outputs for all tasks
        task_outputs = []
        for task_idx in range(self.num_tasks):
            task_output = self.output_layers[task_idx](combined)
            task_outputs.append(task_output)

        output = torch.cat(task_outputs, dim=1)
        return F.sigmoid(output)


class CascadedNN(nn.Module):
    """
    Cascaded/Sequential Neural Network

    Architecture:
    - Shared feature extraction layers
    - Predictions are made sequentially: P(Y ≤ y_i+1) uses P(Y ≤ y_i) as input
    - Natural for CDF estimation since F(y_i) ≤ F(y_i+1)
    - Enforces monotonicity through architecture

    Args:
        input_dim: Input feature dimension
        output_dim: Number of tasks/outputs (threshold locations)
        hidden_dims: List of hidden layer dimensions (default: [128, 64])
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64]):
        super(CascadedNN, self).__init__()
        self.num_tasks = output_dim

        # Shared feature extraction
        self.shared_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Cascaded prediction layers
        # First task: only uses shared features
        self.first_task_layer = nn.Linear(hidden_dims[-1], 1)

        # Subsequent tasks: use shared features + previous prediction
        self.cascade_layers = nn.ModuleList([
            nn.Linear(hidden_dims[-1] + 1, 1) for _ in range(output_dim - 1)
        ])

    def forward(self, x):
        if x.dim() == 1:
            x = x.reshape(-1, 1)

        # Shared feature extraction
        features = x
        for layer in self.shared_layers:
            features = F.relu(layer(features))

        # Cascaded predictions
        task_outputs = []

        # First task prediction
        first_output = torch.sigmoid(self.first_task_layer(features))
        task_outputs.append(first_output)

        # Subsequent tasks use previous predictions
        for task_idx in range(1, self.num_tasks):
            # Concatenate shared features with previous prediction
            prev_prediction = task_outputs[-1]
            combined = torch.cat([features, prev_prediction], dim=1)

            # Predict current task
            task_output = torch.sigmoid(self.cascade_layers[task_idx - 1](combined))

            # Enforce monotonicity: current prediction >= previous prediction
            # Using max to ensure F(y_i+1) >= F(y_i)
            task_output = torch.maximum(task_output, prev_prediction)

            task_outputs.append(task_output)

        # Concatenate all predictions
        output = torch.cat(task_outputs, dim=1)
        return output


# Additional utility: Parameter counting function
def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage and parameter comparison
if __name__ == "__main__":
    # Test with typical dimensions
    input_dim = 20
    output_dim = 19  # 19 quantiles
    batch_size = 32

    # Create test input
    x = torch.randn(batch_size, input_dim)

    # Original SimpleNN from dte_mtl.py
    class SimpleNN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, output_dim)
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return F.sigmoid(self.fc3(x))

    # Test all models
    models = {
        'SimpleNN (Original)': SimpleNN(input_dim, output_dim),
        'MMoE': MMoENN(input_dim, output_dim, num_experts=3, expert_hidden=64),
        'Cross-Stitch': CrossStitchNN(input_dim, output_dim),
        'Cascaded': CascadedNN(input_dim, output_dim)
    }

    print("=" * 70)
    print("Multi-Task Architecture Comparison")
    print("=" * 70)
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim} (number of tasks/thresholds)")
    print(f"Batch size: {batch_size}")
    print("=" * 70)

    for name, model in models.items():
        # Forward pass
        output = model(x)
        params = count_parameters(model)

        print(f"\n{name}:")
        print(f"  Parameters: {params:,}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

        # Check monotonicity for cascaded model
        if 'Cascaded' in name:
            is_monotonic = (output[:, 1:] >= output[:, :-1]).all().item()
            print(f"  Monotonic: {is_monotonic}")

    print("\n" + "=" * 70)
    print("Parameter Comparison:")
    baseline_params = count_parameters(models['SimpleNN (Original)'])
    for name, model in models.items():
        params = count_parameters(model)
        ratio = params / baseline_params
        print(f"  {name:20s}: {params:6,} ({ratio:.2f}x baseline)")
    print("=" * 70)
