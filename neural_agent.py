import random

import torch
import torch.nn as nn
from torch.distributions import Categorical 

from nsfr.utils.common import load_module


class NeuralAgent(nn.Module):
    def __init__(self, env, rules, device, rng=None):
        super(NeuralAgent, self).__init__()

        self.device = device
        self.rng = random.Random() if rng is None else rng
        self.env = env
        
        # Load the environment-specific MLP module
        # This is already optimized for the environment's state representation
        mlp_module_path = f"in/envs/{self.env.name}/mlp.py"
        module = load_module(mlp_module_path)
        
        # Create neural actor network using the environment's MLP implementation
        # We don't use softmax here as we'll apply it in the forward method
        self.actor = module.MLP(
            device=device, 
            has_softmax=False,
            out_size=len(self.env.pred2action.keys()),
            logic=True
        )
        
        # Create critic network for value function
        self.critic = module.MLP(
            device=device, 
            has_softmax=False,
            out_size=1,
            logic=True
        )
        
        # Action space setup (similar to LogicalAgent)
        self.num_actions = len(self.env.pred2action.keys())
        self.env_action_id_to_action_pred_indices = self._build_action_id_dict()
        
        # For exploration (same as LogicalAgent)
        self.uniform = Categorical(
            torch.tensor([1.0 / self.num_actions for _ in range(self.num_actions)], device=device))
        self.upprior = Categorical(
            torch.tensor([0.9] + [0.1 / (self.num_actions-1) for _ in range(self.num_actions-1)], device=device))
    
    def forward(self, x):
        """
        Forward pass through the neural actor
        Returns action probabilities directly
        """
        # Get raw logits from actor network
        action_logits = self.actor(x)
        # Apply softmax to get probabilities
        return torch.softmax(action_logits, dim=-1)
    
    def _print(self):
        """Print method for compatibility with LogicalAgent"""
        print("Neural policy")
    
    def _build_action_id_dict(self):
        """
        Builds a mapping from environment action IDs to predicate indices
        This is similar to LogicalAgent but simpler
        """
        env_action_names = list(self.env.pred2action.keys())
        env_action_id_to_action_pred_indices = {}
        
        # Each environment action maps to its own index
        for i in range(len(env_action_names)):
            env_action_id_to_action_pred_indices[i] = [i]
                
        return env_action_id_to_action_pred_indices
    
    def to_action_distribution(self, raw_action_probs):
        """
        Convert raw action probabilities to environment action distribution
        Simpler than LogicalAgent since we map directly
        """
        # For our neural agent, the raw_action_probs already have the right format
        batch_size = raw_action_probs.size(0)
        
        # Ensure we have the right number of actions (padding if necessary)
        if raw_action_probs.size(1) < self.env.n_raw_actions:
            # Pad with zeros if needed
            padding = torch.zeros(batch_size, 
                                 self.env.n_raw_actions - raw_action_probs.size(1), 
                                 device=self.device)
            action_dist = torch.cat([raw_action_probs, padding], dim=1)
        else:
            # Use as is
            action_dist = raw_action_probs[:, :self.env.n_raw_actions]
            
        return action_dist
    
    def get_action_and_value(self, logic_state, action=None):
        """
        Interface compatible with LogicalAgent.get_action_and_value
        Returns action, log probability, entropy, and value estimate
        """
        # Get action probabilities from neural network
        raw_action_probs = self(logic_state)
        
        # Convert to environment action distribution
        action_dist = self.to_action_distribution(raw_action_probs)
        dist = Categorical(action_dist)
        
        # Sample action if not provided
        if action is None:
            action = dist.sample()
        
        # Get log probabilities and entropy
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # Get value estimate
        value = self.critic(logic_state)
        
        return action, logprob, entropy, value
    
    def get_value(self, logic_state):
        """Get value estimate for a state"""
        return self.critic(logic_state).flatten()
    
    