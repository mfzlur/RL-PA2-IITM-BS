# Temporal Difference Learning: SARSA vs Q-Learning

Implementation of SARSA (on-policy) and Q-Learning (off-policy) algorithms for reinforcement learning assignment.

## Project Structure

```
.
├── env.py                          # Environment creation functions
├── grid_world.py                   # GridWorld environment implementation
├── td_learning_experiments.py      # Main experiment runner
├── visualize_results.py            # Visualization and analysis
├── requirements.txt                # Python dependencies
├── results/                        # Experiment results (generated)
│   ├── *.pkl                      # Serialized results
│   └── *_config.json              # Configuration files
└── plots/                          # Generated visualizations
    ├── training_curves_*.png
    ├── state_visits_*.png
    ├── qvalue_policy_*.png
    └── comparison_*.png
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import numpy, matplotlib, seaborn, plotly; print('All dependencies installed successfully!')"
```

## Usage

### Run All Experiments

This will run all 20 experiments with hyperparameter tuning:

```bash
python td_learning_experiments.py
```
To utilise multiprocessor by parallelizing the hyperparameter tuning runs (seeds) and the 100 final runs
```bash
python td_learning_experiments_parallelized.py
```

**Note:** This will take several hours to complete (~6,400 training runs).

### Generate Visualizations

After experiments complete, generate all plots:

```bash
python visualize_results.py
```

## Experiment Configurations

### 10×10 Grid World (16 configurations)

**Q-Learning (8 configs):**
- Varying: `transition_prob` ∈ {0.7, 1.0}, `start_state` ∈ {(0,4), (3,6)}, exploration ∈ {ε-greedy, Softmax}
- Fixed: `wind=False`

**SARSA (8 configs):**
- Varying: `wind` ∈ {True, False}, `start_state` ∈ {(0,4), (3,6)}, exploration ∈ {ε-greedy, Softmax}
- Fixed: `transition_prob=1.0`

### Four Room Grid World (4 configurations)

**Q-Learning (2 configs):**
- Varying: `goal_change` ∈ {True, False}

**SARSA (2 configs):**
- Varying: `goal_change` ∈ {True, False}

## Hyperparameter Tuning

Each configuration is tuned over:
- Learning rate α ∈ {0.001, 0.01, 0.1, 1.0}
- Discount factor γ ∈ {0.7, 0.8, 0.9, 1.0}
- ε-greedy: ε ∈ {0.001, 0.01, 0.05, 0.1}
- Softmax: τ ∈ {0.01, 0.1, 1.0, 2.0}

Each combination tested with **5 random seeds**.

## Algorithms

### Q-Learning (Off-Policy)
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```
- Uses max over next state actions
- Learns optimal policy regardless of exploration

### SARSA (On-Policy)
```
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
```
- Uses actual next action from policy
- Learns value of policy being followed

## Output Files

### Results Directory
- `*.pkl`: Serialized experiment results containing:
  - Episode rewards and steps
  - Q-tables averaged over 100 runs
  - State visit frequencies
  - Optimal policies and value functions
- `*_config.json`: Human-readable configuration files

### Plots Directory
- `training_curves_*.png`: Reward and step progression over episodes
- `state_visits_*.png`: Heatmaps of state visitation frequencies
- `qvalue_policy_*.png`: Q-value heatmaps with optimal policy arrows
- `comparison_*.png`: Side-by-side SARSA vs Q-Learning comparisons

## Key Implementation Features

1. **Exploration Strategies:**
   - ε-greedy: Random action with probability ε
   - Softmax: Boltzmann exploration with temperature τ

2. **Stochastic Transitions:**
   - Probability p: intended direction
   - Probability (1-p)×0.5: perpendicular directions

3. **Terminal Conditions:**
   - Reach goal state OR exceed 100 timesteps

4. **State Representation:**
   - Sequential indexing (0 to num_states-1)
   - Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

## Performance Metrics

- **Average reward per episode**: Cumulative reward across episode
- **Average steps per episode**: Steps needed to reach goal
- **State visitation frequency**: How often each state is visited
- **Optimal policy**: Best action for each state
- **Value function**: Expected return from each state

## Expected Runtime

- Hyperparameter tuning: ~2-3 hours per configuration
- Final experiments: ~30-60 minutes per configuration
- **Total**: 6-8 hours for all 20 configurations

## Troubleshooting

### Memory Issues
If you run out of memory, reduce `num_runs` in `run_final_experiments()` from 100 to 50.

### Slow Execution
For faster testing, reduce:
- `num_seeds` in `tune_hyperparameters()` from 5 to 3
- `num_episodes` in `ExperimentConfig` from 5000 to 2000

### Visualization Errors
Ensure all experiments completed successfully before running visualizations.

## Citation

If using this code, please cite the original assignment:
```
DA5007: Special Topics in ML (Reinforcement Learning)
Programming Assignment 2: Temporal Difference Learning
```

## License

This code is provided for educational purposes as part of a coursework assignment.
