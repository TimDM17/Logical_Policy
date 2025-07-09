# Import standard Python libraries for file operations, randomness and timing
import os
import random
import time
import pickle
import yaml
from pathlib import Path
from dataclasses import dataclass

# Import numerical and deep learning libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import custom components for neural symbolic reinforcement learning
from logic_agent import LogicalAgent            # Neural-symbolic agent that uses logical reasoning
from env_vectorized import VectorizedNudgeBaseEnv  # Environment wrapper
from utils import save_hyperparams                 # Utility for saving experiment parameters 

# Optional: RTPT for tracking remaining training time
try:
    from rtpt import RTPT
    has_rtpt = True
except ImportError:
    has_rtpt = False
    print("RTPT not found. Progress tracking will be limited.")

@dataclass
class Args:
    # Environment settings
    env_name: str = "seaquest"             # Game environment to uese       
    
    # Training parameters               
    seed: int = 8                       # Random seed for reproducibility   
    num_envs: int = 8                   # Number of parallel environments for data collection   
    num_steps: int = 128                # Steps to collect per environment before updating   
    total_timesteps: int = 10000000     # Total environment steps for the entire training run   
    save_steps: int = 1000000           # Save model checkpoint every million steps   
    
    # Learning hyperparameters
    learning_rate: float = 2.5e-4       # Learning rate for Adam optmizer          
    gamma: float = 0.99                 # Discount factor for future rewards   
    gae_lambda: float = 0.95            # Lambda parameter for GAE advantage estimation   
    update_epochs: int = 4              # Number of passes through the data for each update   
    clip_coef: float = 0.1              # PPO clipping coefficient for trust region   
    ent_coef: float = 0.01              # Entropy coefficient to encourage exploration   
    
    # Rules and setup
    rules: str = "default"              # Rule set to use for logical policy                 
    recover: bool = False               # Whether to recover training from checkpoint   
    mode: str = "logic"                 # Environment mode   
    
    # Computed parameters (set at runtime)
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

# Constants defining the training process
OUT_PATH = Path("out_logic/")            # Root directory for all output files       
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

def load_model_train(checkpoint_dir, n_envs, device):
    """Load most recent checkpoint and training state"""
    # Find all checkpoing files
    checkpoints = list(checkpoint_dir.glob("step_*.pth"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    # Get most recent checkpoing
    steps = [int(c.stem.split('_')[1]) for c in checkpoints]
    most_recent_step = max(steps)
    most_recent_checkpoint = checkpoint_dir / f"step_{most_recent_step}.pth"
    
    print(f"Loading checkpoint from step {most_recent_step}")
    
    # Load the environment to get proper dimensions
    args_path = checkpoint_dir.parent / "config.yaml"
    with open(args_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environnment
    env = VectorizedNudgeBaseEnv.from_name(
        config['env_name'],
        n_envs=n_envs,
        mode="logic",
        seed=config['seed']
    )
    
    # Create agent and load state
    agent = LogicalAgent(env, rules=config['rules'], device=device)
    agent.load_state_dict(torch.load(most_recent_checkpoint, map_location=device))
    agent.to(device)
    
    return agent, most_recent_step

def train(args):
    """
    Main training function that implements PPO algorithm for training the logical agent.
    This function orchestrates the complete training process:
        - Initializes environments and agent
        - Collects experience using the current policy
        - Computes advantages and returns
        - Updates policy and value networks
        - Logs metrics and saves checkpoints
    """
    
    # Calculate derived parameters
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // 4
    args.num_iterations = args.total_timesteps // args.batch_size
    
    # Create directory structure for experiment outputs
    run_name = f"{args.env_name}_logic_lr_{args.learning_rate}_gamma_{args.gamma}_entcoef_{args.ent_coef}"
    experiment_dir = OUT_PATH / "runs" / run_name      # Main experiment directory
    checkpoint_dir = experiment_dir / "checkpoints"    # Directory for model checkpoints
    writer_dir = OUT_PATH / "tensorboard" / run_name   # Directory for TensorBoard logs
    os.makedirs(checkpoint_dir, exist_ok=True)         # Create checkpoint directory if it doesn't exist
    os.makedirs(writer_dir, exist_ok=True)             # Create TensorBoarad directory if it doesn't exist
    
    # Initialize RTPT for progress tracking
    if has_rtpt:
        rtpt = RTPT(name_initials="TS", experiment_name="LogicPolicy",
                   max_iterations=int(args.total_timesteps / args.save_steps))
        rtpt.start()
    
    # Initialize TensorBoard for tracking and visualizing metrics
    writer = SummaryWriter(writer_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Set random seed for reproducibility across all random number sources
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Prepare for training recovery if requested
    if args.recover:
        agent, global_step = load_model_train(checkpoint_dir, n_envs=args.num_envs, device=DEVICE)
        save_step_bar = global_step
        
        # Load training logs
        with open(checkpoint_dir / "training_log.pkl", "rb") as f:
            episodic_returns, episodic_lengths, value_losses, policy_losses, entropies = pickle.load(f)
    else:
        # Create environment and agent from scratch
        envs = VectorizedNudgeBaseEnv.from_name(args.env_name,
                                              n_envs=args.num_envs, 
                                              mode=args.mode,
                                              seed=args.seed)
        agent = LogicalAgent(envs, rules=args.rules, device=DEVICE)
        agent.to(DEVICE)
        
        # Initialize tracking variables
        global_step = 0
        save_step_bar = args.save_steps
        episodic_returns = []
        episodic_lengths = []
        value_losses = []
        policy_losses = []
        entropies = []
    
    # Print initial logical rules and their weights for reference
    print("\n===== INITIAL LOGICAL POLICY =====")
    agent._print()
    
    # Setup optimizer with separate parameter groups for actor and critic
    optimizer = optim.Adam([
        {"params": agent.logic_actor.parameters(), "lr": args.learning_rate}, # Optimize NSFR logical policy  
        {"params": agent.critic.parameters(), "lr": args.learning_rate},      # Optimize value network 
    ], eps=1e-5) # Small epsilon for numerical stability 
    
    # Create storage tensors for collecting experience
    envs = agent.env
    logic_observation_space = (envs.n_objects, 4)     # Shape of object-centric state representation         
    action_space = ()                                 # Shape of actions (scalar, so empty tuple)         
    
    # Initialize tensors for storing trajectory data 
    logic_obs = torch.zeros((args.num_steps, args.num_envs) + logic_observation_space).to(DEVICE)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_space).to(DEVICE)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(DEVICE)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(DEVICE)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(DEVICE)
    values = torch.zeros((args.num_steps, args.num_envs)).to(DEVICE)
    
    # Track rewards per environment
    episodic_game_returns = torch.zeros((args.num_envs)).to(DEVICE)
    
    # Start timer for computing training speed
    start_time = time.time()
    
    # Reset environment to get initial observations
    next_logic_obs, _ = envs.reset()
    next_logic_obs = next_logic_obs.to(DEVICE)
    next_done = torch.zeros(args.num_envs).to(DEVICE)
    
    # Main training loop - iterate through the specified number of update cycles
    for iteration in range(1, args.num_iterations + 1):
        # Collect trajectory data by interacting with environments
        for step in range(args.num_steps):
            global_step += args.num_envs                      # Update step counter
            logic_obs[step] = next_logic_obs                  # Store current observation
            dones[step] = next_done                           # Store current done flags
            
            # Get action from policy without computing gradients (data collection phase)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_logic_obs)
                values[step] = value.flatten()                # Store value estimate
            
            actions[step] = action                            # Store selected action
            logprobs[step] = logprob                          # Store log probability
            
            # Execute action in environment and observe result
            (next_logic_obs_np, _), reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_logic_obs = next_logic_obs_np.float().to(DEVICE) # Convert to tensor and move to device  
            
            # Process termination flags (episode ended due to failure or time limit)
            terminations = np.array(terminations)             # Convert to numpy array
            truncations = np.array(truncations)               # Convert to numpy array
            next_done = np.logical_or(terminations, truncations)  # Combined termination signal
            rewards[step] = torch.tensor(reward).to(DEVICE).view(-1)  # Store rewards
            next_done = torch.Tensor(next_done).to(DEVICE)    # Convert to tensor and move to device
            
            # Track cumulative returns per environment
            episodic_game_returns += torch.tensor(reward).to(DEVICE).view(-1)
            
            # Process episode completions to log statistics
            for idx, info in enumerate(infos):
                if "episode" in info:  # Environment signals episode completion
                    # Print and log episode statistics with additional return tracking
                    print(f"│ Step={global_step:8d} │ Env={idx:2d} │ Return={info['episode']['r']:6.1f} │ Length={info['episode']['l']:4d} │ Game Return={episodic_game_returns[idx]:6.1f} │")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    writer.add_scalar("charts/episodic_game_return", episodic_game_returns[idx], global_step)
                    episodic_returns.append(info["episode"]["r"])  # Store for later analysis
                    episodic_lengths.append(info["episode"]["l"])  # Store for later analysis
                    
                    # Reset the game return for this environment
                    episodic_game_returns[idx] = 0
            
            # Save model checkpoint when reaching milestone
            if global_step > save_step_bar:
                if has_rtpt:
                    rtpt.step()  # Update progress bar
                
                checkpoint_path = checkpoint_dir / f"step_{save_step_bar}.pth"
                torch.save(agent.state_dict(), checkpoint_path)
                print(f"\n✓ Saved model at: {checkpoint_path}\n")
                
                # Save hyperparameters
                save_hyperparams(
                    args=args,
                    save_path=experiment_dir / "config.yaml",
                    print_summary=False
                )
                
                # Save training data
                training_log = (episodic_returns, episodic_lengths, value_losses, policy_losses, entropies)
                with open(checkpoint_dir / "training_log.pkl", "wb") as f:
                    pickle.dump(training_log, f)
                
                # Update next checkpoint target
                save_step_bar += args.save_steps
        
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        with torch.no_grad():  # No gradients needed for this computation
            next_value = agent.get_value(next_logic_obs).reshape(1, -1)  # Get value of final state
            advantages = torch.zeros_like(rewards).to(DEVICE)  # Initialize advantage tensor
            lastgaelam = 0  # Running GAE value
            
            # Compute GAE in reverse order (from last step to first)
            for t in reversed(range(args.num_steps)):
                # Handle final step differently
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done   # Mask for non-terminal states  
                    nextvalues = next_value             # Use final value estimate
                else:
                    nextnonterminal = 1.0 - dones[t + 1]  # Mask for non-terminal states
                    nextvalues = values[t + 1]            # Use stored value estimates
                
                # GAE formula components
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]  # TD error  
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam  # GAE update
            
            returns = advantages + values  # Compute returns (advantage + value baseline)
        
        # Prepare data for optmization by flattening batch dimensions
        b_logic_obs = logic_obs.reshape((-1,) + logic_observation_space)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_space)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # Prepare indices for minibatch sampling
        b_inds = np.arange(args.batch_size)
        clipfracs = []  # Track fraction of clipped updates for minitoring
        
        # Policy optimization phase - multiple epochs over the collected data
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)  # Shuffle data for each epoch
            
            # Process data in minibatches
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end] # Indices for current minibatch  
                
                # Compute updated probabilites and values
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_logic_obs[mb_inds], b_actions.long()[mb_inds]
                )
                
                # Compute probability ratio for PPO
                logratio = newlogprob - b_logprobs[mb_inds] # Difference in log probabilites  
                ratio = logratio.exp()                      # Probability ratio 
                
                # Calculate KL divergence for monitoring
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()  # Approximate KL divergence
                    approx_kl = ((ratio - 1) - logratio).mean()  # Alternative formulation
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()] # Track clipping  
                
                # Normalize advantages for more stable training
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Compute PPO clipped surrogate objective for policy loss
                pg_loss1 = -mb_advantages * ratio  # Unclipped surrogate objective
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) # Clipped surrogate  
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # Take the maximum (pessimistic bound)
                
                # Compute value loss with clipping for stability
                newvalue = newvalue.view(-1)  # Flatten value predictions
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2  # Unclipped MSE loss
                
                # Implement clipped value loss similar to PPO's policy clipping
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)  # Take the maximum (pessimistic)
                v_loss = 0.5 * v_loss_max.mean()  # Scale value loss
                
                # Compute entropy loss to encourage exploration
                entropy_loss = entropy.mean()
                
                # Combine losses with appropriate weightings
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * 0.5
                
                # Perform gradient update
                optimizer.zero_grad()  # Clear previous gradients
                loss.backward()        # Compute gradients
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5) # Clip gradients for stability  
                optimizer.step()       # Apply gradient update
        
        # Compute explained variance metric (quality of value function)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # Print iteration summary
        sps = int(global_step / (time.time() - start_time))
        print(f"\n=== Iteration {iteration}/{args.num_iterations} | Step {global_step} | SPS: {sps} ===")
        
        if iteration % 10 == 0:
            # Print recent statistics
            avg_return = np.mean(episodic_returns[-10:]) if episodic_returns else 0
            avg_length = np.mean(episodic_lengths[-10:]) if episodic_lengths else 0
            print(f"Recent Stats | Return: {avg_return:.1f} | Length: {avg_length:.1f}")
        
        # Log all metrics to TensorBoard
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        
        # Store metrics for later analysis
        value_losses.append(v_loss.item())
        policy_losses.append(pg_loss.item())
        entropies.append(entropy_loss.item())
        
        # Print current weights of logical rules to monitor learning
        print("\n===== CURRENT LOGICAL POLICY =====")
        agent._print()
    
    # Final save at end of training
    checkpoint_path = checkpoint_dir / f"step_final_{global_step}.pth"
    torch.save(agent.state_dict(), checkpoint_path)
    print(f"\n✓ Saved final model at: {checkpoint_path}")
    
    # Save final training data
    training_log = (episodic_returns, episodic_lengths, value_losses, policy_losses, entropies)
    with open(checkpoint_dir / "training_log_final.pkl", "wb") as f:
        pickle.dump(training_log, f)
    
    # Cleanup: close environment and TensorBoard writer
    envs.close()
    writer.close()

# Entry point: run the training function when script is executed
if __name__ == "__main__":
    # Parse arguments and run training
    import tyro
    args = tyro.cli(Args)
    train(args)