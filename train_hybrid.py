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
from hybrid_agent import HybridAgent        # Hybrid agent combining logical and neural policies
from env_common.env_vectorized import VectorizedNudgeBaseEnv # Environment wrapper
from utils import save_hyperparams          # Utility for saving experiment parameters

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
    env_name: str = "seaquest"            # Game environment to use       
    
    # Training parameters               
    seed: int = 8                         # Random seed for reproducibility   
    num_envs: int = 8                     # Number of parallel environments for data collection   
    num_steps: int = 128                  # Steps to collect per environment before updating   
    total_timesteps: int = 10000000       # Total environment steps for the entire training run   
    save_steps: int = 1000000             # Save model checkpoint every million steps   
    
    # Learning hyperparameters
    learning_rate: float = 2.5e-4         # Learning rate for optimization   
    gamma: float = 0.99                   # Discount factor for future rewards   
    gae_lambda: float = 0.95              # GAE lambda parameter   
    clip_coef: float = 0.1                # PPO clipping coefficient for trust region  
    ent_coef: float = 0.01                # Entropy coefficient for exploration   
    update_epochs: int = 4                # Epochs to process each batch of experience   
    
    # Model configuration
    mode: str = "logic"                   # Environment observation mode (logic for object-centric)
    rules: str = "default"                # Rule set for logical policy
    
    # KL divergence hyperparameter
    kl_coef: float = 0.5                  # Weight for KL divergence loss
    
    # Recovery options
    recover: bool = False                 # Whether to recover from a previous run

# Setup constants and paths
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_PATH = Path("out_hybrid")             # Output directory for hybrid training
OUT_PATH.mkdir(exist_ok=True)             # Create output directory if needed


def load_model_train(checkpoint_dir, n_envs, device):
    """Load a previously saved model for continued training"""
    # Find most recent checkpoint
    checkpoints = list(checkpoint_dir.glob("step_*.pth"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
        
    # Sort by step number and get latest
    checkpoints = sorted(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    most_recent_checkpoint = checkpoints[-1]
    most_recent_step = int(most_recent_checkpoint.stem.split('_')[-1])
    
    # Load configuration from saved YAML
    config_file = checkpoint_dir.parent / "config.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment based on saved config
    env = VectorizedNudgeBaseEnv.from_name(
        config['env_name'],
        n_envs=n_envs,
        mode="logic",
        seed=config['seed']
    )
    
    # Create hybrid agent and load state
    agent = HybridAgent(env, rules=config['rules'], device=device, kl_coef=config.get('kl_coef', 0.5))
    agent.load_state_dict(torch.load(most_recent_checkpoint, map_location=device))
    agent.to(device)
    
    return agent, most_recent_step


def train(args):
    """
    Main training function that implements PPO algorithm for training the hybrid agent.
    This function orchestrates the complete training process:
        - Initializes environments and hybrid agent
        - Collects experience using the neural policy
        - Computes advantages, returns and KL divergence
        - Updates neural policy to optimize both RL objectives and KL matching
        - Logs metrics and saves checkpoints
    """

    # Calculate derived parameters
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // 4
    args.num_iterations = args.total_timesteps // args.batch_size

    # Setup TensorBoard, RTPT, output dirs
    run_name = f"{args.env_name}_hybrid_kl{args.kl_coef}_{args.rules}_seed_{args.seed}"
    experiment_dir = OUT_PATH / "runs" / run_name      # Main experiment directory
    checkpoint_dir = experiment_dir / "checkpoints"    # Directory for model checkpoints
    writer_dir = OUT_PATH / "tensorboard" / run_name   # Directory for TensorBoard logs

    # Create directories if needed
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer_dir.mkdir(parents=True, exist_ok=True)

    # Setup TensorBoard writer
    writer = SummaryWriter(writer_dir)
    writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

    # Save hyperparameters
    save_hyperparams(args, experiment_dir / "config.yaml")

    # Configure RTPT for progress tracking if available
    if has_rtpt:
        rtpt = RTPT(
            name_initials='HYB',
            experiment_name=f"Hybrid_{args.env_name}",
            max_iterations=args.num_iterations
        )
        rtpt.start()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Starting state
    global_step = 0
    save_step_bar = args.save_steps
    episodic_returns = []
    episodic_lengths = []
    value_losses = []
    policy_losses = []
    kl_losses = []

    # Resume from checkpoint or start new training
    if args.recover:
        print("\n Recovering from previous training...")
        agent, global_step = load_model_train(checkpoint_dir, args.num_envs, DEVICE)
        envs = agent.env
        
        # Adjust tracking variables
        save_step_bar = ((global_step // args.save_steps) + 1) * args.save_steps
        print(f"Recovered from step {global_step}, next save at {save_step_bar}")
    else:
        print("\n Starting new training...")
        # Initialize environment and agent
        envs = VectorizedNudgeBaseEnv.from_name(args.env_name,
                                              n_envs=args.num_envs, 
                                              mode=args.mode,
                                              seed=args.seed)
        agent = HybridAgent(envs, rules=args.rules, device=DEVICE, kl_coef=args.kl_coef)
        agent.to(DEVICE)
    
    # Policy printing after agent initialization
    print("\n===== INITIAL HYBRID AGENT POLICY =====")
    agent._print()
    
    # Initialize optimizer (only for neural agent parameters)
    optimizer = optim.Adam([
        {"params": agent.neural_agent.parameters(), "lr": args.learning_rate},
    ], eps=1e-5)

    # Define obervation space for object-centric representation
    logic_observation_space = (envs.n_objects, 4) # Shape of object-centric state representation

    # Initialize experience buffers
    logic_obs = torch.zeros((args.num_steps, args.num_envs) + logic_observation_space,
                           dtype=torch.float32, device=DEVICE)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=DEVICE)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=DEVICE)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=DEVICE)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=DEVICE)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=DEVICE)
    kl_divs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=DEVICE)

    # Initialize tracking of episode returns
    episodic_game_returns = torch.zeros((args.num_envs)).to(DEVICE)

    # Get initial observations
    next_logic_obs_np, _ = envs.reset() 
    next_logic_obs = torch.tensor(next_logic_obs_np).float().to(DEVICE)
    next_done = torch.zeros(args.num_envs, device=DEVICE)

    print(f"\n Starting training for {args.total_timesteps} steps...\n")

    # Main training loop
    for iteration in range(1, args.num_iterations + 1):
        # Progress tracking
        start_time = time.time()
        
        # Data collection phase
        for step in range(args.num_steps):
            global_step += args.num_envs
            
            # Store current observations
            logic_obs[step] = next_logic_obs
            dones[step] = next_done
            
            # Get action from hybrid agent (also computes KL)
            with torch.no_grad():
                action, logprob, _, value, kl_div, _ = agent.get_action_and_value_with_kl(next_logic_obs)
                values[step] = value.flatten()                # Store value estimate
                kl_divs[step] = kl_div                        # Store KL divergence
            
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
                    print(f"│ Step={global_step:8d} │ Env={idx:2d} │ Return={info['episode']['r']:6.1f} │ Length={info['episode']['l']:4d} │ Game Return={episodic_game_returns[idx]:6.1f} │ KL={kl_div.item():6.4f} │")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    writer.add_scalar("charts/episodic_game_return", episodic_game_returns[idx], global_step)
                    writer.add_scalar("charts/step_kl_divergence", kl_div.item(), global_step)
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
                
                # Update next milestone
                save_step_bar += args.save_steps
        
        # Advantage estimation phase
        with torch.no_grad():
            # Calculate value estimate for last state
            next_value = agent.get_value(next_logic_obs)
            
            # Initialize advantage and return estimation
            advantages = torch.zeros_like(rewards).to(DEVICE)
            lastgaelam = 0
            
            # Compute GAE (Generalized Advantage Estimation)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                # Compute TD error and GAE
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            
            # Compute returns for value function targets (advantages + values)
            returns = advantages + values
        
        # Flatten batch data for minibatch processing
        b_logic_obs = logic_obs.reshape((-1,) + logic_observation_space)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_kl_divs = kl_divs.reshape(-1)
        
        # Policy update phase
        # Track metrics for this iteration
        pg_losses = []
        value_losses = []
        entropy_losses = []
        kl_loss_values = []
        
        # Update policy over multiple epochs
        for epoch in range(args.update_epochs):
            # Generate random indices for minibatch sampling
            indices = np.random.permutation(args.batch_size)
            
            # Process data in minibatches
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = indices[start:end]
                
                # Get predictions for minibatch data
                _, newlogprob, entropy, newvalue, kl_div, kl_loss = agent.get_action_and_value_with_kl(
                    b_logic_obs[mb_inds], b_actions.long()[mb_inds]
                )
                
                # Compute surrogate loss terms
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                # Normalize advantages within batch for more stable learning
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Compute policy gradient loss with PPO clipping
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Compute value function loss with clipping
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                
                # Compute entropy loss (for exploration)
                entropy_loss = entropy.mean()
                
                # Compute total loss (including KL divergence) !!!!!!!!!!!!!!!!!!!
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * 0.5 + kl_loss
                
                # Track losses
                pg_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                kl_loss_values.append(kl_loss.item())
                
                # Gradient update
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.neural_agent.parameters(), 0.5)  # Gradient clipping
                optimizer.step()
        
        # Log metrics for this iteration
        avg_pg_loss = np.mean(pg_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy_loss = np.mean(entropy_losses)
        avg_kl_loss = np.mean(kl_loss_values)
        avg_kl_div = torch.mean(b_kl_divs).item()
        
        policy_losses.append(avg_pg_loss)
        value_losses.append(avg_value_loss)
        
        # Calculate explained variance (measure of value function accuracy)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # Print iteration summary
        print(f"│ Iteration={iteration:4d}/{args.num_iterations} │ Step={global_step:8d} │", end=" ")
        print(f"Policy Loss={avg_pg_loss:.4f} │ Value Loss={avg_value_loss:.4f} │", end=" ")
        print(f"KL Loss={avg_kl_loss:.4f} │ KL Div={avg_kl_div:.4f} │", end=" ")
        print(f"Entropy={avg_entropy_loss:.4f} │ Time={time.time() - start_time:.2f}s │")
        
        # Write metrics to TensorBoard
        writer.add_scalar("losses/policy_loss", avg_pg_loss, global_step)
        writer.add_scalar("losses/value_loss", avg_value_loss, global_step)
        writer.add_scalar("losses/kl_loss", avg_kl_loss, global_step)
        writer.add_scalar("losses/entropy", avg_entropy_loss, global_step)
        writer.add_scalar("metrics/kl_divergence", avg_kl_div, global_step)
        writer.add_scalar("metrics/explained_variance", explained_var, global_step)
    
    # Final save and cleanup
    final_checkpoint_path = checkpoint_dir / f"step_final_{global_step}.pth"
    torch.save(agent.state_dict(), final_checkpoint_path)
    print(f"\n✓ Saved final model at: {final_checkpoint_path}")
    
    # Save training statistics
    stats = {
        "episodic_returns": episodic_returns,
        "episodic_lengths": episodic_lengths,
        "value_losses": value_losses,
        "policy_losses": policy_losses,
        "kl_losses": kl_loss_values
    }
    with open(experiment_dir / "training_log_final.pkl", "wb") as f:
        pickle.dump(stats, f)
    
    # Close environments and writer
    envs.close()
    writer.close()

if __name__ == "__main__":
    # Parse arguments and run training
    import tyro
    args = tyro.cli(Args)
    train(args)

