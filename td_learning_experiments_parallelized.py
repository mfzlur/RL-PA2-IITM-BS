import numpy as np
import pickle
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, replace
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from env import create_standard_grid, create_four_room
from grid_world import seq_to_col_row

# --- Multiprocessing Imports ---
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
# -------------------------------


class ExplorationStrategy:
    """Base class for exploration strategies"""

    def select_action(self, Q: np.ndarray, state: int, num_actions: int) -> int:
        raise NotImplementedError

class EpsilonGreedy(ExplorationStrategy):
    """Epsilon-greedy exploration strategy"""

    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def select_action(self, Q: np.ndarray, state: int, num_actions: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(num_actions)
        else:
            # Break ties randomly
            max_q = np.max(Q[state, :])
            max_actions = np.where(Q[state, :] == max_q)[0]
            return np.random.choice(max_actions)

class Softmax(ExplorationStrategy):
    """Softmax (Boltzmann) exploration strategy"""

    def __init__(self, temperature: float):
        self.temperature = temperature

    def select_action(self, Q: np.ndarray, state: int, num_actions: int) -> int:
        # Subtract max for numerical stability
        q_values = Q[state, :]
        q_max = np.max(q_values)
        exp_q = np.exp((q_values - q_max) / self.temperature)
        probs = exp_q / np.sum(exp_q)

        # Handle potential numerical issues
        probs = np.clip(probs, 0, 1)
        probs = probs / np.sum(probs)

        return np.random.choice(num_actions, p=probs)

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    env_type: str  # 'standard' or 'four_room'
    algorithm: str  # 'q_learning' or 'sarsa'
    start_state: Tuple[int, int]
    transition_prob: float
    wind: bool
    goal_change: bool
    exploration_type: str  # 'epsilon_greedy' or 'softmax'
    alpha: float  # learning rate
    gamma: float  # discount factor
    epsilon: float = 0.1  # for epsilon-greedy
    temperature: float = 1.0  # for softmax
    num_episodes: int = 5000
    max_steps: int = 100
    seed: int = 42

class TDLearningAgent:
    """Base class for TD learning algorithms"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.Q = None
        self.state_visits = None
        self.episode_rewards = []
        self.episode_steps = []

    def initialize_q_table(self, num_states: int, num_actions: int):
        """Initialize Q-table with zeros"""
        self.Q = np.zeros((num_states, num_actions))
        self.state_visits = np.zeros((num_states,))

    def get_exploration_strategy(self) -> ExplorationStrategy:
        """Get the exploration strategy based on config"""
        if self.config.exploration_type == 'epsilon_greedy':
            return EpsilonGreedy(self.config.epsilon)
        elif self.config.exploration_type == 'softmax':
            return Softmax(self.config.temperature)
        else:
            raise ValueError(f"Unknown exploration type: {self.config.exploration_type}")

    def train(self, env) -> Dict[str, Any]:
        """Train the agent - to be implemented by subclasses"""
        raise NotImplementedError

    def get_optimal_policy(self) -> np.ndarray:
        """Extract optimal policy from Q-table"""
        return np.argmax(self.Q, axis=1)

    def get_value_function(self) -> np.ndarray:
        """Extract value function from Q-table"""
        return np.max(self.Q, axis=1)

class QLearningAgent(TDLearningAgent):
    """Q-Learning (off-policy TD control) agent"""

    def train(self, env) -> Dict[str, Any]:
        """Train using Q-Learning algorithm"""
        np.random.seed(self.config.seed)

        num_states = env.num_states
        num_actions = env.num_actions

        self.initialize_q_table(num_states, num_actions)
        exploration = self.get_exploration_strategy()

        self.episode_rewards = []
        self.episode_steps = []

        for episode in range(self.config.num_episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0

            for step in range(self.config.max_steps):
                # Select action using exploration strategy
                action = exploration.select_action(self.Q, state, num_actions)

                # Take action
                next_state, reward = env.step(state, action)
                reward = reward.item()
                episode_reward += reward

                # Track state visits
                self.state_visits[state] += 1

                # Q-Learning update: uses max over next actions (off-policy)
                max_next_q = np.max(self.Q[next_state, :])

                # Check if terminal state
                is_terminal = next_state in env.goal_states_seq

                if is_terminal:
                    td_target = reward
                else:
                    td_target = reward + self.config.gamma * max_next_q

                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.config.alpha * td_error

                state = next_state
                steps += 1

                if is_terminal:
                    break

            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(steps)

        return {
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'final_q_table': self.Q.copy(),
            'state_visits': self.state_visits.copy(),
            'optimal_policy': self.get_optimal_policy(),
            'value_function': self.get_value_function()
        }

class SARSAAgent(TDLearningAgent):
    """SARSA (on-policy TD control) agent"""

    def train(self, env) -> Dict[str, Any]:
        """Train using SARSA algorithm"""
        np.random.seed(self.config.seed)

        num_states = env.num_states
        num_actions = env.num_actions

        self.initialize_q_table(num_states, num_actions)
        exploration = self.get_exploration_strategy()

        self.episode_rewards = []
        self.episode_steps = []

        for episode in range(self.config.num_episodes):
            state = env.reset()
            action = exploration.select_action(self.Q, state, num_actions)

            episode_reward = 0
            steps = 0

            for step in range(self.config.max_steps):
                # Take action
                next_state, reward = env.step(state, action)
                reward = reward.item()
                episode_reward += reward

                # Track state visits
                self.state_visits[state] += 1

                # Check if terminal state
                is_terminal = next_state in env.goal_states_seq

                if is_terminal:
                    # Terminal state update
                    td_target = reward
                    td_error = td_target - self.Q[state, action]
                    self.Q[state, action] += self.config.alpha * td_error
                    steps += 1
                    break
                else:
                    # SARSA update: uses next action from policy (on-policy)
                    next_action = exploration.select_action(self.Q, next_state, num_actions)

                    td_target = reward + self.config.gamma * self.Q[next_state, next_action]
                    td_error = td_target - self.Q[state, action]
                    self.Q[state, action] += self.config.alpha * td_error

                    state = next_state
                    action = next_action
                    steps += 1

            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(steps)

        return {
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'final_q_table': self.Q.copy(),
            'state_visits': self.state_visits.copy(),
            'optimal_policy': self.get_optimal_policy(),
            'value_function': self.get_value_function()
        }

def create_experiment_configs() -> List[ExperimentConfig]:
    """Create all 20 experiment configurations as specified in the assignment"""
    configs = []

    # 10x10 Grid World - Q-Learning (8 configurations)
    # Varying: transition_prob (0.7, 1.0), start_state ((0,4), (3,6)), exploration (epsilon, softmax)
    # Fixed: wind=False
    for trans_prob in [0.7, 1.0]:
        for start_state in [(0, 4), (3, 6)]:
            for explore_type in ['epsilon_greedy', 'softmax']:
                config = ExperimentConfig(
                    env_type='standard',
                    algorithm='q_learning',
                    start_state=start_state,
                    transition_prob=trans_prob,
                    wind=False,
                    goal_change=False,
                    exploration_type=explore_type,
                    alpha=0.1,  # Default, will be tuned
                    gamma=0.9,  # Default, will be tuned
                    epsilon=0.1 if explore_type == 'epsilon_greedy' else 0.0,
                    temperature=1.0 if explore_type == 'softmax' else 0.0
                )
                configs.append(config)

    # 10x10 Grid World - SARSA (8 configurations)
    # Varying: wind (True, False), start_state ((0,4), (3,6)), exploration (epsilon, softmax)
    # Fixed: transition_prob=1.0
    for wind in [True, False]:
        for start_state in [(0, 4), (3, 6)]:
            for explore_type in ['epsilon_greedy', 'softmax']:
                config = ExperimentConfig(
                    env_type='standard',
                    algorithm='sarsa',
                    start_state=start_state,
                    transition_prob=1.0,
                    wind=wind,
                    goal_change=False,
                    exploration_type=explore_type,
                    alpha=0.1,  # Default, will be tuned
                    gamma=0.9,  # Default, will be tuned
                    epsilon=0.1 if explore_type == 'epsilon_greedy' else 0.0,
                    temperature=1.0 if explore_type == 'softmax' else 0.0
                )
                configs.append(config)

    # Four Room - Q-Learning (2 configurations)
    # Varying: goal_change (True, False)
    for goal_change in [True, False]:
        config = ExperimentConfig(
            env_type='four_room',
            algorithm='q_learning',
            start_state=(8, 0),
            transition_prob=1.0,
            wind=False,
            goal_change=goal_change,
            exploration_type='epsilon_greedy',
            alpha=0.1,  # Default, will be tuned
            gamma=0.9,  # Default, will be tuned
            epsilon=0.1
        )
        configs.append(config)

    # Four Room - SARSA (2 configurations)
    # Varying: goal_change (True, False)
    for goal_change in [True, False]:
        config = ExperimentConfig(
            env_type='four_room',
            algorithm='sarsa',
            start_state=(8, 0),
            transition_prob=1.0,
            wind=False,
            goal_change=goal_change,
            exploration_type='epsilon_greedy',
            alpha=0.1,  # Default, will be tuned
            gamma=0.9,  # Default, will be tuned
            epsilon=0.1
        )
        configs.append(config)

    return configs

def get_hyperparameter_grid() -> Dict[str, List[float]]:
    """Get hyperparameter search grid"""
    return {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'gamma': [0.7, 0.8, 0.9, 1.0],
        'epsilon': [0.001, 0.01, 0.05, 0.1],
        'temperature': [0.01, 0.1, 1.0, 2.0]
    }

# --- PARALLEL WORKER FUNCTION ---
# This MUST be a top-level function for multiprocessing to work
def run_single_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Worker function to run a single training instance.
    Creates its own env and agent.
    """
    # Create environment
    if config.env_type == 'standard':
        env = create_standard_grid(
            start_state=np.array([config.start_state]),
            transition_prob=config.transition_prob,
            wind=config.wind
        )
    else:
        env = create_four_room(
            start_state=np.array([config.start_state]),
            goal_change=config.goal_change,
            transition_prob=config.transition_prob
        )

    # Train agent
    if config.algorithm == 'q_learning':
        agent = QLearningAgent(config)
    else:
        agent = SARSAAgent(config)

    results = agent.train(env)
    return results
# ---------------------------------

def tune_hyperparameters(config: ExperimentConfig, num_seeds: int = 5, num_cores: int = 1) -> Tuple[ExperimentConfig, float]:
    """
    Tune hyperparameters for a given configuration in parallel
    Returns best config and best average reward
    """
    hp_grid = get_hyperparameter_grid()

    # Determine which hyperparameters to tune based on exploration strategy
    alphas = hp_grid['alpha']
    gammas = hp_grid['gamma']

    if config.exploration_type == 'epsilon_greedy':
        exploration_params = hp_grid['epsilon']
        param_name = 'epsilon'
    else:
        exploration_params = hp_grid['temperature']
        param_name = 'temperature'

    print(f"\nTuning {config.algorithm} on {config.env_type} with {config.exploration_type}")
    print(f"Config: start={config.start_state}, trans_prob={config.transition_prob}, wind={config.wind}, goal_change={config.goal_change}")

    # 1. Create a list of all 320 (64x5) tasks to run
    tasks_to_run = []
    for alpha in alphas:
        for gamma in gammas:
            for explore_param in exploration_params:
                for seed in range(num_seeds):
                    test_config = replace(
                        config,
                        alpha=alpha,
                        gamma=gamma,
                        epsilon=explore_param if param_name == 'epsilon' else config.epsilon,
                        temperature=explore_param if param_name == 'temperature' else config.temperature,
                        seed=seed
                    )
                    tasks_to_run.append(test_config)

    total_combinations = len(alphas) * len(gammas) * len(exploration_params)
    print(f"Created {len(tasks_to_run)} tuning jobs for {total_combinations} combinations.")
    print(f"Running in parallel on {num_cores} cores...")

    # 2. Run all tasks in parallel using the pool
    all_results = []
    with Pool(processes=num_cores) as pool:
        # Use tqdm for a progress bar
        all_results = list(tqdm(pool.imap(run_single_experiment, tasks_to_run), total=len(tasks_to_run), desc="Tuning"))

    print("Tuning runs complete. Aggregating results...")

    # 3. Aggregate results
    # Group rewards by their hyperparameter combination
    grouped_rewards = {}  # Key: (alpha, gamma, explore_param), Value: list of rewards
    for task_config, result in zip(tasks_to_run, all_results):
        if param_name == 'epsilon':
            explore_val = task_config.epsilon
        else:
            explore_val = task_config.temperature
        
        hp_key = (task_config.alpha, task_config.gamma, explore_val)

        if hp_key not in grouped_rewards:
            grouped_rewards[hp_key] = []
        
        # Use average reward over last 500 episodes as performance metric
        avg_reward = np.mean(result['episode_rewards'][-500:])
        grouped_rewards[hp_key].append(avg_reward)

    # 4. Find the best combination
    best_avg_reward = -np.inf
    best_hp_key = None
    best_config = None

    for hp_key, rewards_list in grouped_rewards.items():
        mean_reward = np.mean(rewards_list)

        if mean_reward > best_avg_reward:
            best_avg_reward = mean_reward
            best_hp_key = hp_key

    # Re-create the best_config object to return
    best_alpha, best_gamma, best_explore_param = best_hp_key
    best_config = replace(
        config,
        alpha=best_alpha,
        gamma=best_gamma,
        epsilon=best_explore_param if param_name == 'epsilon' else config.epsilon,
        temperature=best_explore_param if param_name == 'temperature' else config.temperature
    )

    print(f"Best hyperparameters: α={best_config.alpha}, γ={best_config.gamma}, ", end="")
    if param_name == 'epsilon':
        print(f"ε={best_config.epsilon}")
    else:
        print(f"τ={best_config.temperature}")
    print(f"Best average reward: {best_avg_reward:.2f}\n")

    return best_config, best_avg_reward

def run_final_experiments(config: ExperimentConfig, num_runs: int = 100, num_cores: int = 1) -> Dict[str, Any]:
    """
    Run final experiments with best hyperparameters across multiple runs in parallel
    """
    print(f"Running final experiments for {config.algorithm} on {config.env_type}...")

    # 1. Create a list of 100 tasks to run (one for each seed)
    tasks_to_run = []
    for run in range(num_runs):
        # Use dataclasses.replace to create a copy with the new seed
        run_config = replace(config, seed=run)
        tasks_to_run.append(run_config)

    print(f"Created {len(tasks_to_run)} final runs.")
    print(f"Running in parallel on {num_cores} cores...")

    # 2. Run all 100 tasks in parallel using the pool
    all_results = []
    with Pool(processes=num_cores) as pool:
        all_results = list(tqdm(pool.imap(run_single_experiment, tasks_to_run), total=len(tasks_to_run), desc="Final Runs", leave=True))

    print("Final runs complete. Aggregating results...")

    # 3. Aggregate results from the list
    all_rewards = []
    all_steps = []
    all_state_visits = []
    all_q_tables = []

    for results in all_results:
        all_rewards.append(results['episode_rewards'])
        all_steps.append(results['episode_steps'])
        all_state_visits.append(results['state_visits'])
        all_q_tables.append(results['final_q_table'])

    # Aggregate results
    all_rewards = np.array(all_rewards)
    all_steps = np.array(all_steps)
    all_state_visits = np.array(all_state_visits)
    all_q_tables = np.array(all_q_tables)

    aggregated_results = {
        'mean_rewards_per_episode': np.mean(all_rewards, axis=0),
        'std_rewards_per_episode': np.std(all_rewards, axis=0),
        'mean_steps_per_episode': np.mean(all_steps, axis=0),
        'std_steps_per_episode': np.std(all_steps, axis=0),
        'mean_state_visits': np.mean(all_state_visits, axis=0),
        'mean_q_table': np.mean(all_q_tables, axis=0),
        'optimal_policy': np.argmax(np.mean(all_q_tables, axis=0), axis=1),
        'value_function': np.max(np.mean(all_q_tables, axis=0), axis=1),
        'config': config,
        'num_runs': num_runs
    }

    return aggregated_results

def save_results(results: Dict[str, Any], config: ExperimentConfig, output_dir: str = 'results'):
    """Save experiment results"""
    Path(output_dir).mkdir(exist_ok=True)

    # Create filename
    filename = f"{config.env_type}_{config.algorithm}_{config.start_state[0]}_{config.start_state[1]}_"
    filename += f"tp{config.transition_prob}_w{config.wind}_gc{config.goal_change}_{config.exploration_type}"

    # Save as pickle
    with open(f"{output_dir}/{filename}.pkl", 'wb') as f:
        pickle.dump(results, f)

    # Save config as JSON
    config_dict = {
        'env_type': config.env_type,
        'algorithm': config.algorithm,
        'start_state': config.start_state,
        'transition_prob': config.transition_prob,
        'wind': config.wind,
        'goal_change': config.goal_change,
        'exploration_type': config.exploration_type,
        'alpha': config.alpha,
        'gamma': config.gamma,
        'epsilon': config.epsilon,
        'temperature': config.temperature
    }

    with open(f"{output_dir}/{filename}_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

def main():
    """Main execution function"""
    # --- Determine number of cores to use ---
    try:
        # Get os.cpu_count() if available
        num_cores = os.cpu_count()
    except NotImplementedError:
        # Fallback for systems where this isn't available
        num_cores = 4 
    
    # You can also manually set this, e.g., num_cores = 20
    # num_cores = 20 
    
    print("="*80)
    print("TD Learning Experiments: SARSA vs Q-Learning (Parallelized)")
    print(f"--- Using {num_cores} CPU cores ---")
    print("="*80)

    # Create all experiment configurations
    configs = create_experiment_configs()
    print(f"\nCreated {len(configs)} experiment configurations")

    # Part A: Hyperparameter Tuning
    print("\n" + "="*80)
    print("PART A: Hyperparameter Tuning")
    print("="*80)

    tuned_configs = []
    for i, config in enumerate(configs, 1):
        print(f"\n[Experiment {i}/{len(configs)}]")
        best_config, best_reward = tune_hyperparameters(config, num_seeds=5, num_cores=num_cores)
        tuned_configs.append(best_config)

    # Part B: Final Experiments with Best Hyperparameters
    print("\n" + "="*80)
    print("PART B: Final Experiments (100 runs each)")
    print("="*80)

    all_results = []
    for i, config in enumerate(tuned_configs, 1):
        print(f"\n[Experiment {i}/{len(tuned_configs)}]")
        results = run_final_experiments(config, num_runs=100, num_cores=num_cores)
        all_results.append(results)
        save_results(results, config)

    print("\n" + "="*80)
    print("All experiments completed!")
    print(f"Results saved to '{os.path.abspath('results/')}' directory")
    print("="*80)

    return all_results

if __name__ == "__main__":
    # This check is CRITICAL for multiprocessing to work
    results = main()