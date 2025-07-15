import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from logic_agent import LogicalAgent
from neural_agent import NeuralAgent

class HybridAgent(nn.Module):
    """
    Hybrid agent that combines logical and neural policies with KL divergence.
    
    This agent wraps both a logical agent (using NSFR) and a neural agent.
    During training, it computes KL divergence between their action distributions,
    which can be used as an additional loss term to make the neural policy mimic
    the logical policy's behavior.
    """
    
    def __init__(self, env, rules, device, kl_coef=0.5, rng=None):
        """
        Initialize the hybrid agent.
        
        Args:
            env: Environment instance
            rules: Name of ruleset to use for logical agent
            device: Device to run computations on
            kl_coef: Weight for KL divergence loss
            rng: Random number generator
        """
        super(HybridAgent, self).__init__()
        self.device = device
        self.kl_coef = kl_coef
        
        # Initialize both agents
        self.logical_agent = LogicalAgent(env, rules, device, rng)
        self.neural_agent = NeuralAgent(env, rules, device, rng)
        
        # Freeze logical agent parameters (we only want to train the neural agent)
        for param in self.logical_agent.parameters():
            param.requires_grad = False
            
        # Environment info
        self.env = env
        
    def forward(self, x):
        """
        Forward pass computes both logical and neural action probabilities.
        
        Args:
            x: State tensor
            
        Returns:
            logical_dist: Action distribution from logical agent
            neural_dist: Action distribution from neural agent
        """
        # Run both agents
        logical_raw_probs = self.logical_agent.logic_actor(x)
        logical_dist = self.logical_agent.to_action_distribution(logical_raw_probs)
        
        neural_raw_probs = self.neural_agent(x)
        neural_dist = self.neural_agent.to_action_distribution(neural_raw_probs)
        
        return logical_dist, neural_dist
    
    def compute_kl_divergence(self, logical_dist, neural_dist):
        """
        Compute KL divergence from neural to logical distribution.
        
        Args:
            logical_dist: Action distribution from logical agent
            neural_dist: Action distribution from neural agent
            
        Returns:
            kl_div: KL(neural_dist || logical_dist)
        """
        # Add small epsilon to prevent division by zero or log of zero
        epsilon = 1e-8
        logical_dist = logical_dist + epsilon
        neural_dist = neural_dist + epsilon
        
        # Normalize distributions
        logical_dist = logical_dist / logical_dist.sum(dim=-1, keepdim=True)
        neural_dist = neural_dist / neural_dist.sum(dim=-1, keepdim=True)
        
        # Calculate KL divergence: KL(neural || logical)
        # This pushes neural to match logical
        kl_div = F.kl_div(
            neural_dist.log(), 
            logical_dist, 
            reduction='batchmean',
            log_target=False
        )
        
        return kl_div
    
    def get_action_and_value(self, state, action=None):
        """
        Standard interface - uses neural agent but doesn't compute KL.
        
        Args:
            state: Environment state
            action: Optional action to evaluate
            
        Returns:
            Same as neural_agent.get_action_and_value
        """
        return self.neural_agent.get_action_and_value(state, action)
    
    def get_action_and_value_with_kl(self, state, action=None):
        """
        Extended interface that also computes KL divergence.
        
        Args:
            state: Environment state
            action: Optional action to evaluate
            
        Returns:
            action: Selected action
            logprob: Log probability of the action
            entropy: Entropy of the neural policy
            value: Value estimate from neural critic
            kl_div: KL divergence from neural to logical policy
            kl_loss: Weighted KL divergence (kl_coef * kl_div)
        """
        # Get logical and neural distributions
        logical_dist, neural_dist = self(state)
        
        # Create distribution for neural policy
        dist = Categorical(neural_dist)
        
        # Sample action if not provided
        if action is None:
            action = dist.sample()
        
        # Get log probabilities and value from neural agent
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.neural_agent.critic(state)
        
        # Compute KL divergence
        kl_div = self.compute_kl_divergence(logical_dist, neural_dist)
        kl_loss = self.kl_coef * kl_div
        
        return action, logprob, entropy, value, kl_div, kl_loss
    
    def get_value(self, state):
        """Get value estimate for a state from neural critic"""
        return self.neural_agent.get_value(state)
    
    def _print(self):
        """Print both agents' policies"""
        print("==== Hybrid Agent ====")
        print("Logical component:")
        self.logical_agent._print()
        print("\nNeural component:")
        self.neural_agent._print()