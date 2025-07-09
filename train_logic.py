# Import standard Python libraries for file operations, randomness, and timing
import os
import random
import time
import pickle
from pathlib import Path

# Import numerical and deep learning libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import custom components for neural-symbolic reinforcement learning
from logic_agent import LogicalAgent            # Neural-symbolic agent that uses logical reasoning
from env_vectorized import VectorizedNudgeBaseEnv  # Environment wrapper that provides standardized interface
from utils import save_hyperparams               # Utility for saving experiment parameters

# Constants defining the training process
OUT_PATH = Path("out_logic/")                   # Root directory for all output files
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
NUM_ENVS = 8                                    # Number of parallel environments for data collection
NUM_STEPS = 128                                 # Steps to collect per environment before updating
TOTAL_TIMESTEPS = 10000000                      # Total environment steps for the entire training run
SAVE_STEPS = 1000000                            # Save model checkpoint every million steps
BATCH_SIZE = NUM_ENVS * NUM_STEPS               # Total transitions collected before update (1024)
MINIBATCH_SIZE = BATCH_SIZE // 4                # Size of minibatches for gradient updates (256)
NUM_ITERATIONS = TOTAL_TIMESTEPS // BATCH_SIZE  # Number of complete update cycles
LEARNING_RATE = 2.5e-4                          # Learning rate for Adam optimizer
GAMMA = 0.99                                    # Discount factor for future rewards
GAE_LAMBDA = 0.95                               # Lambda parameter for GAE advantage estimation
UPDATE_EPOCHS = 4                               # Number of passes through the data for each update
CLIP_COEF = 0.1                                 # PPO clipping coefficient for trust region
ENT_COEF = 0.01                                 # Entropy coefficient to encourage exploration
SEED = 0                                        # Random seed for reproducibility

def train():
    """
    Main training function that implements PPO algorithm for training the logical agent.
    This function orchestrates the complete training process:
    - Initializes environments and agent
    - Collects experience using the current policy
    - Computes advantages and returns
    - Updates policy and value networks
    - Logs metrics and saves checkpoints
    """
    
    # Create directory structure for experiment outputs
    run_name = f"donkeykong_logic_lr_{LEARNING_RATE}"  # Unique name for this training run
    experiment_dir = OUT_PATH / "runs" / run_name      # Main experiment directory
    checkpoint_dir = experiment_dir / "checkpoints"    # Directory for model checkpoints
    writer_dir = OUT_PATH / "tensorboard" / run_name   # Directory for TensorBoard logs
    os.makedirs(checkpoint_dir, exist_ok=True)         # Create checkpoint directory if it doesn't exist
    os.makedirs(writer_dir, exist_ok=True)             # Create TensorBoard directory if it doesn't exist
    
    # Initialize TensorBoard for tracking and visualizing metrics
    writer = SummaryWriter(writer_dir)
    
    # Set random seeds for reproducibility across all random number sources
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Create environment and agent
    env_name = "alien"                                         # Game environment to use
    envs = VectorizedNudgeBaseEnv.from_name(env_name,          # Create 8 parallel environments
                                          n_envs=NUM_ENVS, 
                                          mode="logic",         # Use logical mode (vs. 'ppo' mode)
                                          seed=SEED)
    agent = LogicalAgent(envs, rules="default", device=DEVICE)  # Create logical agent with default rules
    agent.to(DEVICE)                                           # Move agent to GPU if available
    
    # Print initial logical rules and their weights for reference
    agent._print()
    
    # Setup optimizer with separate parameter groups for actor and critic
    optimizer = optim.Adam([
        {"params": agent.logic_actor.parameters(), "lr": LEARNING_RATE},  # Optimize NSFR logical policy
        {"params": agent.critic.parameters(), "lr": LEARNING_RATE},       # Optimize value network
    ], eps=1e-5)  # Small epsilon for numerical stability
    
    # Create storage tensors for collecting experience
    logic_observation_space = (envs.n_objects, 4)              # Shape of object-centric state representation
    action_space = ()                                          # Shape of actions (scalar, so empty tuple)
    # Initialize tensors for storing trajectory data
    logic_obs = torch.zeros((NUM_STEPS, NUM_ENVS) + logic_observation_space).to(DEVICE)  # States
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + action_space).to(DEVICE)               # Actions
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)                            # Log probabilities
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)                             # Rewards
    dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)                               # Episode terminations
    values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)                              # Value estimates
    
    # Initialize tracking variables for monitoring training progress
    global_step = 0                  # Total environment steps taken
    save_step_bar = SAVE_STEPS       # Next step count for saving checkpoint
    episodic_returns = []            # Track episode returns for statistics
    episodic_lengths = []            # Track episode lengths for statistics
    value_losses = []                # Track value network losses
    policy_losses = []               # Track policy network losses
    entropies = []                   # Track policy entropy values
    
    # Start timer for computing training speed
    start_time = time.time()
    
    # Reset environment to get initial observations
    _, next_logic_obs = envs.reset()                      # Get initial logical state
    next_logic_obs = next_logic_obs.to(DEVICE)            # Move to device
    next_done = torch.zeros(NUM_ENVS).to(DEVICE)          # Initialize done flags
    
    # Main training loop - iterate through the specified number of update cycles
    for iteration in range(1, NUM_ITERATIONS + 1):
        # Collect trajectory data by interacting with environments
        for step in range(NUM_STEPS):
            global_step += NUM_ENVS                        # Update step counter
            logic_obs[step] = next_logic_obs               # Store current observation
            dones[step] = next_done                        # Store current done flags
            
            # Get action from policy without computing gradients (data collection phase)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_logic_obs)
                values[step] = value.flatten()             # Store value estimate
            
            actions[step] = action                         # Store selected action
            logprobs[step] = logprob                       # Store log probability
            
            # Execute action in environment and observe result
            _, next_logic_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_logic_obs = next_logic_obs_np.float().to(DEVICE)  # Convert to tensor and move to device
            
            # Process termination flags (episode ended due to failure or time limit)
            terminations = np.array(terminations)          # Convert to numpy array
            truncations = np.array(truncations)            # Convert to numpy array
            next_done = np.logical_or(terminations, truncations)  # Combined termination signal
            rewards[step] = torch.tensor(reward).to(DEVICE).view(-1)  # Store rewards
            next_done = torch.Tensor(next_done).to(DEVICE)  # Convert to tensor and move to device
            
            # Process episode completions to log statistics
            for idx, info in enumerate(infos):
                if "episode" in info:  # Environment signals episode completion
                    # Print and log episode statistics
                    print(f"global_step={global_step}, env={idx}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    episodic_returns.append(info["episode"]["r"])  # Store for later analysis
                    episodic_lengths.append(info["episode"]["l"])  # Store for later analysis
            
            # Save model checkpoint when reaching milestone
            if global_step > save_step_bar:
                checkpoint_path = checkpoint_dir / f"step_{save_step_bar}.pth"
                torch.save(agent.state_dict(), checkpoint_path)
                print(f"\nSaved model at: {checkpoint_path}")
                save_step_bar += SAVE_STEPS  # Update next checkpoint target
        
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        with torch.no_grad():  # No gradients needed for this computation
            next_value = agent.get_value(next_logic_obs).reshape(1, -1)  # Get value of final state
            advantages = torch.zeros_like(rewards).to(DEVICE)  # Initialize advantage tensor
            lastgaelam = 0  # Running GAE value
            
            # Compute GAE in reverse order (from last step to first)
            for t in reversed(range(NUM_STEPS)):
                # Handle final step differently
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done  # Mask for non-terminal states
                    nextvalues = next_value            # Use final value estimate
                else:
                    nextnonterminal = 1.0 - dones[t + 1]  # Mask for non-terminal states
                    nextvalues = values[t + 1]            # Use stored value estimates
                
                # GAE formula components
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]  # TD error
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam  # GAE update
            
            returns = advantages + values  # Compute returns (advantage + value baseline)
        
        # Prepare data for optimization by flattening batch dimensions
        b_logic_obs = logic_obs.reshape((-1,) + logic_observation_space)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_space)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # Prepare indices for minibatch sampling
        b_inds = np.arange(BATCH_SIZE)
        clipfracs = []  # Track fraction of clipped updates for monitoring
        
        # Policy optimization phase - multiple epochs over the collected data
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)  # Shuffle data for each epoch
            
            # Process data in minibatches
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]  # Indices for current minibatch
                
                # Compute updated probabilities and values
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_logic_obs[mb_inds], b_actions.long()[mb_inds]
                )
                
                # Compute probability ratio for PPO
                logratio = newlogprob - b_logprobs[mb_inds]  # Difference in log probabilities
                ratio = logratio.exp()                       # Probability ratio
                
                # Calculate KL divergence for monitoring
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()  # Approximate KL divergence
                    approx_kl = ((ratio - 1) - logratio).mean()  # Alternative formulation
                    clipfracs += [((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()]  # Track clipping
                
                # Normalize advantages for more stable training
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Compute PPO clipped surrogate objective for policy loss
                pg_loss1 = -mb_advantages * ratio  # Unclipped surrogate objective
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)  # Clipped surrogate
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # Take the maximum (pessimistic bound)
                
                # Compute value loss with clipping for stability
                newvalue = newvalue.view(-1)  # Flatten value predictions
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2  # Unclipped MSE loss
                
                # Implement clipped value loss similar to PPO's policy clipping
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -CLIP_COEF,
                    CLIP_COEF,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)  # Take the maximum (pessimistic)
                v_loss = 0.5 * v_loss_max.mean()  # Scale value loss
                
                # Compute entropy loss to encourage exploration
                entropy_loss = entropy.mean()
                
                # Combine losses with appropriate weightings
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * 0.5
                
                # Perform gradient update
                optimizer.zero_grad()  # Clear previous gradients
                loss.backward()        # Compute gradients
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)  # Clip gradients for stability
                optimizer.step()       # Apply gradient update
        
        # Compute explained variance metric (quality of value function)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # Log all metrics to TensorBoard
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        # Store metrics for later analysis
        value_losses.append(v_loss.item())
        policy_losses.append(pg_loss.item())
        entropies.append(entropy_loss.item())
        
        # Print current weights of logical rules to monitor learning
        agent._print()
    
    # Cleanup: close environment and TensorBoard writer
    envs.close()
    writer.close()

# Entry point: run the training function when script is executed
if __name__ == "__main__":
    train()