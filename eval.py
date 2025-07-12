import numpy as np
import torch
import os
from logic_agent import LogicalAgent
from env_vectorized import VectorizedNudgeBaseEnv

def evaluate_agent(model_path, n_episodes=10, verbose=False):
    """Evaluate a trained agent and report performance metrics"""
    # Setup
    print(f"\nEvaluating model: {os.path.basename(model_path)}")
    print("-" * 50)
    
    # Initialize environment and agent
    env = VectorizedNudgeBaseEnv.from_name("seaquest", n_envs=1, mode="logic", seed=8)
    agent = LogicalAgent(env, rules="default1", device="cpu")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return []
        
    agent.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    agent.eval()
    
    # Track results
    returns = []
    lengths = []
    
    # Evaluate episodes
    for i in range(n_episodes):
        logic_obs, _ = env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        
        # Run episode
        while not done:
            episode_length += 1
            
            # Get action from policy
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(torch.tensor(logic_obs).float())
            
            # Execute action
            (logic_obs, _), reward, terminated, truncated, info = env.step(action.numpy())
            done = terminated[0] or truncated[0]
            total_reward += reward[0]
            
            # Safety check
            if episode_length >= 10000:
                if verbose:
                    print("Episode exceeded 10000 steps, terminating")
                done = True
        
        # Record results
        returns.append(total_reward)
        lengths.append(episode_length)
        print(f"Episode {i+1:2d}: Return = {total_reward:4.1f}, Length = {episode_length:4d}")
    
    # Summary statistics
    print("-" * 50)
    print(f"Average return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Average length: {np.mean(lengths):.1f}")
    print(f"Min/Max return: {min(returns):.1f}/{max(returns):.1f}")
    
    return returns

# Run evaluation
if __name__ == "__main__":
    model_path = "out_logic/runs/seaquest_default1_seed_8/checkpoints/step_final_9999360.pth"
    evaluate_agent(model_path)