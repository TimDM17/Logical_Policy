
import os
import random
import time
import pickle
import yaml
from pathlib import Path
from dataclasses import dataclass


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from logic_agent import LogicalAgent            
from env_vectorized import VectorizedNudgeBaseEnv  
from utils import save_hyperparams               


try:
    from rtpt import RTPT
    has_rtpt = True
except ImportError:
    has_rtpt = False
    print("RTPT not found. Progress tracking will be limited.")

@dataclass
class Args:
    
    env_name: str = "alien"                
    
    
    seed: int = 0                          
    num_envs: int = 8                      
    num_steps: int = 128                   
    total_timesteps: int = 10000000        
    save_steps: int = 1000000              
    
    
    learning_rate: float = 2.5e-4          
    gamma: float = 0.99                    
    gae_lambda: float = 0.95               
    update_epochs: int = 4                 
    clip_coef: float = 0.1                 
    ent_coef: float = 0.01                 
    
    
    rules: str = "default"                 
    recover: bool = False                  
    mode: str = "logic"                    
    
    
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


OUT_PATH = Path("out_logic/")                   
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

def load_model_train(checkpoint_dir, n_envs, device):
    
    
    checkpoints = list(checkpoint_dir.glob("step_*.pth"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    
    steps = [int(c.stem.split('_')[1]) for c in checkpoints]
    most_recent_step = max(steps)
    most_recent_checkpoint = checkpoint_dir / f"step_{most_recent_step}.pth"
    
    print(f"Loading checkpoint from step {most_recent_step}")
    
    
    args_path = checkpoint_dir.parent / "config.yaml"
    with open(args_path, 'r') as f:
        config = yaml.safe_load(f)
    
    
    env = VectorizedNudgeBaseEnv.from_name(
        config['env_name'],
        n_envs=n_envs,
        mode="logic",
        seed=config['seed']
    )
    
    
    agent = LogicalAgent(env, rules=config['rules'], device=device)
    agent.load_state_dict(torch.load(most_recent_checkpoint, map_location=device))
    agent.to(device)
    
    return agent, most_recent_step

def train(args):
    
    
    
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // 4
    args.num_iterations = args.total_timesteps // args.batch_size
    
    
    run_name = f"{args.env_name}_logic_lr_{args.learning_rate}_gamma_{args.gamma}_entcoef_{args.ent_coef}"
    experiment_dir = OUT_PATH / "runs" / run_name      
    checkpoint_dir = experiment_dir / "checkpoints"    
    writer_dir = OUT_PATH / "tensorboard" / run_name   
    os.makedirs(checkpoint_dir, exist_ok=True)         
    os.makedirs(writer_dir, exist_ok=True)             
    
    
    if has_rtpt:
        rtpt = RTPT(name_initials="TS", experiment_name="LogicPolicy",
                   max_iterations=int(args.total_timesteps / args.save_steps))
        rtpt.start()
    
    
    writer = SummaryWriter(writer_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    
    if args.recover:
        agent, global_step = load_model_train(checkpoint_dir, n_envs=args.num_envs, device=DEVICE)
        save_step_bar = global_step
        
        
        with open(checkpoint_dir / "training_log.pkl", "rb") as f:
            episodic_returns, episodic_lengths, value_losses, policy_losses, entropies = pickle.load(f)
    else:
        
        envs = VectorizedNudgeBaseEnv.from_name(args.env_name,
                                              n_envs=args.num_envs, 
                                              mode=args.mode,
                                              seed=args.seed)
        agent = LogicalAgent(envs, rules=args.rules, device=DEVICE)
        agent.to(DEVICE)
        
        
        global_step = 0
        save_step_bar = args.save_steps
        episodic_returns = []
        episodic_lengths = []
        value_losses = []
        policy_losses = []
        entropies = []
    
    
    print("\n===== INITIAL LOGICAL POLICY =====")
    agent._print()
    
   
    optimizer = optim.Adam([
        {"params": agent.logic_actor.parameters(), "lr": args.learning_rate},  
        {"params": agent.critic.parameters(), "lr": args.learning_rate},       
    ], eps=1e-5)  
    
    
    envs = agent.env
    logic_observation_space = (envs.n_objects, 4)              
    action_space = ()                                          
    
    
    logic_obs = torch.zeros((args.num_steps, args.num_envs) + logic_observation_space).to(DEVICE)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_space).to(DEVICE)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(DEVICE)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(DEVICE)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(DEVICE)
    values = torch.zeros((args.num_steps, args.num_envs)).to(DEVICE)
    
    
    episodic_game_returns = torch.zeros((args.num_envs)).to(DEVICE)
    
    
    start_time = time.time()
    
    
    next_logic_obs, _ = envs.reset()
    next_logic_obs = next_logic_obs.to(DEVICE)
    next_done = torch.zeros(args.num_envs).to(DEVICE)
    
    
    for iteration in range(1, args.num_iterations + 1):
        
        for step in range(args.num_steps):
            global_step += args.num_envs                      
            logic_obs[step] = next_logic_obs                  
            dones[step] = next_done                           
            
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_logic_obs)
                values[step] = value.flatten()                
            
            actions[step] = action                            
            logprobs[step] = logprob                          
            
            
            (next_logic_obs_np, _), reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_logic_obs = next_logic_obs_np.float().to(DEVICE)  
            
            
            terminations = np.array(terminations)             
            truncations = np.array(truncations)               
            next_done = np.logical_or(terminations, truncations)  
            rewards[step] = torch.tensor(reward).to(DEVICE).view(-1)  
            next_done = torch.Tensor(next_done).to(DEVICE)    
            
            
            episodic_game_returns += torch.tensor(reward).to(DEVICE).view(-1)
            
            
            for idx, info in enumerate(infos):
                if "episode" in info:  
                   
                    print(f"│ Step={global_step:8d} │ Env={idx:2d} │ Return={info['episode']['r']:6.1f} │ Length={info['episode']['l']:4d} │ Game Return={episodic_game_returns[idx]:6.1f} │")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    writer.add_scalar("charts/episodic_game_return", episodic_game_returns[idx], global_step)
                    episodic_returns.append(info["episode"]["r"])  
                    episodic_lengths.append(info["episode"]["l"])  
                    
                    
                    episodic_game_returns[idx] = 0
            
            
            if global_step > save_step_bar:
                if has_rtpt:
                    rtpt.step()  
                
                checkpoint_path = checkpoint_dir / f"step_{save_step_bar}.pth"
                torch.save(agent.state_dict(), checkpoint_path)
                print(f"\n✓ Saved model at: {checkpoint_path}\n")
                
                
                save_hyperparams(
                    args=args,
                    save_path=experiment_dir / "config.yaml",
                    print_summary=False
                )
                
                
                training_log = (episodic_returns, episodic_lengths, value_losses, policy_losses, entropies)
                with open(checkpoint_dir / "training_log.pkl", "wb") as f:
                    pickle.dump(training_log, f)
                
                
                save_step_bar += args.save_steps
        
        
        with torch.no_grad():  
            next_value = agent.get_value(next_logic_obs).reshape(1, -1)  
            advantages = torch.zeros_like(rewards).to(DEVICE)  
            lastgaelam = 0  
            
            
            for t in reversed(range(args.num_steps)):
                
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done  
                    nextvalues = next_value            
                else:
                    nextnonterminal = 1.0 - dones[t + 1]  
                    nextvalues = values[t + 1]            
                
                
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]  
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam  
            
            returns = advantages + values  
        
        
        b_logic_obs = logic_obs.reshape((-1,) + logic_observation_space)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_space)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        
        b_inds = np.arange(args.batch_size)
        clipfracs = []  
        
        
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)  
            
            
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]  
                
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_logic_obs[mb_inds], b_actions.long()[mb_inds]
                )
                
                
                logratio = newlogprob - b_logprobs[mb_inds]  
                ratio = logratio.exp()                       
                
                
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()  
                    approx_kl = ((ratio - 1) - logratio).mean()  
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]  
                
                
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                
                pg_loss1 = -mb_advantages * ratio  
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)  
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  
                
                
                newvalue = newvalue.view(-1)  
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2  
                
                
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)  
                v_loss = 0.5 * v_loss_max.mean()  
                
                
                entropy_loss = entropy.mean()
                
                
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * 0.5
                
                
                optimizer.zero_grad()  
                loss.backward()        
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)  
                optimizer.step()       
        
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        
        sps = int(global_step / (time.time() - start_time))
        print(f"\n=== Iteration {iteration}/{args.num_iterations} | Step {global_step} | SPS: {sps} ===")
        
        if iteration % 10 == 0:
            
            avg_return = np.mean(episodic_returns[-10:]) if episodic_returns else 0
            avg_length = np.mean(episodic_lengths[-10:]) if episodic_lengths else 0
            print(f"Recent Stats | Return: {avg_return:.1f} | Length: {avg_length:.1f}")
        
        
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        
        
        value_losses.append(v_loss.item())
        policy_losses.append(pg_loss.item())
        entropies.append(entropy_loss.item())
        
        
        print("\n===== CURRENT LOGICAL POLICY =====")
        agent._print()
    
    
    checkpoint_path = checkpoint_dir / f"step_final_{global_step}.pth"
    torch.save(agent.state_dict(), checkpoint_path)
    print(f"\n✓ Saved final model at: {checkpoint_path}")
    
    
    training_log = (episodic_returns, episodic_lengths, value_losses, policy_losses, entropies)
    with open(checkpoint_dir / "training_log_final.pkl", "wb") as f:
        pickle.dump(training_log, f)
    
    
    envs.close()
    writer.close()


if __name__ == "__main__":
    
    import tyro
    args = tyro.cli(Args)
    train(args)