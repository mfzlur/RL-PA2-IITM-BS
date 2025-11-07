
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List
from grid_world import seq_to_col_row
from td_learning_experiments_parallelized import ExperimentConfig

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results(results_dir: str = 'results') -> List[Dict[str, Any]]:
    """Load all experiment results"""
    results = []
    results_path = Path(results_dir)

    for pkl_file in sorted(results_path.glob('*.pkl')):
        with open(pkl_file, 'rb') as f:
            result = pickle.load(f)
            results.append(result)

    return results

def plot_training_curves(results: Dict[str, Any], save_dir: str = 'plots'):
    """Plot training curves: average reward and steps per episode"""
    Path(save_dir).mkdir(exist_ok=True)

    config = results['config']
    mean_rewards = results['mean_rewards_per_episode']
    std_rewards = results['std_rewards_per_episode']
    mean_steps = results['mean_steps_per_episode']
    std_steps = results['std_steps_per_episode']

    episodes = np.arange(len(mean_rewards))

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Average Reward per Episode
    axes[0].plot(episodes, mean_rewards, linewidth=2, label='Mean Reward')
    axes[0].fill_between(episodes, 
                          mean_rewards - std_rewards, 
                          mean_rewards + std_rewards, 
                          alpha=0.3, label='±1 Std Dev')
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Average Reward', fontsize=12)
    axes[0].set_title(f'Average Reward per Episode\n{config.algorithm.upper()} - {config.env_type}', 
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Average Steps to Goal per Episode
    axes[1].plot(episodes, mean_steps, linewidth=2, label='Mean Steps', color='orange')
    axes[1].fill_between(episodes, 
                          mean_steps - std_steps, 
                          mean_steps + std_steps, 
                          alpha=0.3, color='orange', label='±1 Std Dev')
    axes[1].set_xlabel('Episode', fontsize=12)
    axes[1].set_ylabel('Average Steps', fontsize=12)
    axes[1].set_title(f'Average Steps per Episode\n{config.algorithm.upper()} - {config.env_type}', 
                      fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    filename = f"training_curves_{config.env_type}_{config.algorithm}_{config.start_state[0]}_{config.start_state[1]}"
    filename += f"_tp{config.transition_prob}_w{config.wind}_gc{config.goal_change}_{config.exploration_type}.png"
    plt.savefig(f"{save_dir}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved training curves: {filename}")

def plot_state_visit_heatmap(results: Dict[str, Any], save_dir: str = 'plots'):
    """Plot state visit heatmap"""
    Path(save_dir).mkdir(exist_ok=True)

    config = results['config']
    state_visits = results['mean_state_visits']

    # Determine grid dimensions
    if config.env_type == 'standard':
        num_rows, num_cols = 10, 10
    else:
        num_rows, num_cols = 9, 9

    # Reshape to grid
    visit_grid = np.zeros((num_rows, num_cols))
    for state in range(len(state_visits)):
        row_col = seq_to_col_row(state, num_cols)
        row, col = row_col[0, 0], row_col[0, 1]
        visit_grid[row, col] = state_visits[state]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(visit_grid, annot=True, fmt='.0f', cmap='YlOrRd', 
                cbar_kws={'label': 'Average Visits'}, ax=ax, linewidths=0.5)

    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_title(f'State Visit Heatmap\n{config.algorithm.upper()} - {config.env_type}', 
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure
    filename = f"state_visits_{config.env_type}_{config.algorithm}_{config.start_state[0]}_{config.start_state[1]}"
    filename += f"_tp{config.transition_prob}_w{config.wind}_gc{config.goal_change}_{config.exploration_type}.png"
    plt.savefig(f"{save_dir}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved state visit heatmap: {filename}")

def plot_q_value_and_policy(results: Dict[str, Any], save_dir: str = 'plots'):
    """Plot Q-value heatmap with optimal policy overlay"""
    Path(save_dir).mkdir(exist_ok=True)

    config = results['config']
    value_function = results['value_function']
    optimal_policy = results['optimal_policy']

    # Determine grid dimensions
    if config.env_type == 'standard':
        num_rows, num_cols = 10, 10
    else:
        num_rows, num_cols = 9, 9

    # Reshape to grid
    value_grid = np.zeros((num_rows, num_cols))
    policy_grid = np.zeros((num_rows, num_cols), dtype=int)

    for state in range(len(value_function)):
        row_col = seq_to_col_row(state, num_cols)
        row, col = row_col[0, 0], row_col[0, 1]
        value_grid[row, col] = value_function[state]
        policy_grid[row, col] = optimal_policy[state]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot value function as heatmap
    sns.heatmap(value_grid, annot=True, fmt='.1f', cmap='viridis', 
                cbar_kws={'label': 'Max Q-Value'}, ax=ax, linewidths=0.5)

    # Overlay policy arrows
    action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    for row in range(num_rows):
        for col in range(num_cols):
            state = row * num_cols + col
            action = optimal_policy[state]
            symbol = action_symbols[action]

            # Add arrow
            ax.text(col + 0.5, row + 0.7, symbol, 
                   ha='center', va='center', fontsize=16, 
                   color='white', fontweight='bold')

    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_title(f'Q-Value Heatmap with Optimal Policy\n{config.algorithm.upper()} - {config.env_type}', 
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure
    filename = f"qvalue_policy_{config.env_type}_{config.algorithm}_{config.start_state[0]}_{config.start_state[1]}"
    filename += f"_tp{config.transition_prob}_w{config.wind}_gc{config.goal_change}_{config.exploration_type}.png"
    plt.savefig(f"{save_dir}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved Q-value and policy plot: {filename}")

def generate_all_plots(results_dir: str = 'results', save_dir: str = 'plots'):
    """Generate all plots for all experiments"""
    print("\nGenerating visualizations...")
    print("="*80)

    results_list = load_results(results_dir)

    if len(results_list) == 0:
        print("No results found! Please run experiments first.")
        return

    print(f"Found {len(results_list)} experiment results\n")

    for i, results in enumerate(results_list, 1):
        config = results['config']
        print(f"[{i}/{len(results_list)}] Plotting {config.algorithm} - {config.env_type}")

        plot_training_curves(results, save_dir)
        plot_state_visit_heatmap(results, save_dir)
        plot_q_value_and_policy(results, save_dir)
        print()

    print("="*80)
    print(f"All plots saved to '{save_dir}/' directory")
    print("="*80)

def compare_algorithms(results_dir: str = 'results', save_dir: str = 'plots'):
    """Create comparison plots between SARSA and Q-Learning"""
    Path(save_dir).mkdir(exist_ok=True)

    results_list = load_results(results_dir)

    if len(results_list) == 0:
        return

    # Group by configuration (excluding algorithm)
    grouped = {}
    for result in results_list:
        config = result['config']
        key = (config.env_type, config.start_state, config.transition_prob, 
               config.wind, config.goal_change, config.exploration_type)

        if key not in grouped:
            grouped[key] = {}

        grouped[key][config.algorithm] = result

    # Create comparison plots
    print("\nCreating comparison plots...")

    for key, algorithms in grouped.items():
        if len(algorithms) != 2:
            continue

        env_type, start_state, trans_prob, wind, goal_change, explore_type = key

        q_result = algorithms.get('q_learning')
        s_result = algorithms.get('sarsa')

        if q_result is None or s_result is None:
            continue

        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Reward comparison
        episodes = np.arange(len(q_result['mean_rewards_per_episode']))
        axes[0].plot(episodes, q_result['mean_rewards_per_episode'], 
                    linewidth=2, label='Q-Learning', alpha=0.8)
        axes[0].plot(episodes, s_result['mean_rewards_per_episode'], 
                    linewidth=2, label='SARSA', alpha=0.8)
        axes[0].set_xlabel('Episode', fontsize=12)
        axes[0].set_ylabel('Average Reward', fontsize=12)
        axes[0].set_title(f'Reward Comparison\n{env_type}', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Steps comparison
        axes[1].plot(episodes, q_result['mean_steps_per_episode'], 
                    linewidth=2, label='Q-Learning', alpha=0.8)
        axes[1].plot(episodes, s_result['mean_steps_per_episode'], 
                    linewidth=2, label='SARSA', alpha=0.8)
        axes[1].set_xlabel('Episode', fontsize=12)
        axes[1].set_ylabel('Average Steps', fontsize=12)
        axes[1].set_title(f'Steps Comparison\n{env_type}', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = f"comparison_{env_type}_{start_state[0]}_{start_state[1]}"
        filename += f"_tp{trans_prob}_w{wind}_gc{goal_change}_{explore_type}.png"
        plt.savefig(f"{save_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved comparison: {filename}")

if __name__ == "__main__":
    # Generate all individual plots
    generate_all_plots()

    # Generate comparison plots
    compare_algorithms()
