# Logical Policy Implementation

This repository implements a neural-symbolic reinforcement learning approach that combines logical reasoning with deep reinforcement learning. Inspired by the [BlendRL Framework](https://github.com/ml-research/blendrl-dev), this implementation focuses specifically only on using logical rules to guide policy learning in Atari games.

## Overview

The Logical Policy approach uses object-centric representations and logical rules to make the agent's decision process interpretable and structured. This implementation:

- Uses Neural-Symbolic Forward Reasoner (NSFR) for logical reasoning
- Implements PPO (Proximal Policy Optimization) for training
- Supports various Atari environments (Seaquest, DonkeyKong, etc.)
- Allows customization of logical rules

## Installation 
```bash
# Clone the repository
git clone https://github.com/TimDM17/Logical_Policy.git
cd Logical_Policy

# Install dependencies
pip install -r requirements.txt
```

## Training an Agent

Training a logical policy agent is straightforward:

```bash
# Basic training with default settings
python train_logic.py --env_name=seaquest

# Training with custom settings
python train_logic.py --env_name=donkeykong --seed=3 --rules=default --total_timesteps=5000000
```

### Key Training Parameters 
- `--env_name`: Environment to train in (seaquest, donkeykong, alient, etc.)
- `--rules`: Rule set name to use from the environment's rule folder (in/envs/)
- `--seed`: Random seed for reproducibility (important for rule selection)
- `--num_envs`: Number of parallel environments (defualt: 8)
- `--total_timesteps`: Total training steps (default: 10,000,000)

Training progress will be displayed in the console, showing:

- Current iteration and step count
- Episode returns and lengths
- Current logical policy rules and weights

## Evaluating a Trained Agent

Once training is complete, evaluate your agent with:

```bash
python eval.py --model_path=...
```

The evaluation will show per-episode performance statistics and overall averages.

## Customize Rule Sets

### Rule Set Structure

Each environment has rule sets located in
`in/envs/{env_name}/logic/{ruleset_name}/`:

- `clauses.txt`: Contains logical rules in Prolog-like syntax
- `preds.txt`: Action predicates
- `neural_preds.txt`: Predicates for neural module output
- `consts.txt`: Constants used in rules
- `bk.txt`: Background knowledge

### Creating a Custom Rule Set

1. Create a new folder: `in/envs/seaquest/logic/my_custom_rules/`

2. Copy and modify files from an existing rule set

3. Edit `clauses.txt` to define your own rules:

4. Train with your custom rules:
```bash
python train_logic.py --env_name=seaquest --rules=my_custom_rules 
```

### Rule Syntax

Rules follow a Prolog-like syntax:
```
action(X):-condition(A,B), condition2(C).
```

Where:

- `action(X)` is the action to take
- Conditions after `:-` must be satisfied
- Multiple conditions are combined with commas (logical AND)
- Variables like `X`,`A`,`B` are automatically bound

## Monitoring Training

Training progess can be monitored with TensorBoard:
```bash
tensorboard --logdir=out_logic/tensorboard
```

This will show training metrics including:

- Episode returns
- Policy loss
- Value loss
- Entropy
- Learning rate

### Tips 

- Increase exploration with `--ent_coef=0.05` for more challenging environments
- For memory-intensive environments like Alien, reudce `--num_envs=4`
- Use `--recover=True` to continue training from the latest checkpoint

## References

This implementation draws inspiration from:
- [BlendRL: A Framwork for Neural-Symbolic Reinforcement Learning](https://github.com/ml-research/blendrl-dev)