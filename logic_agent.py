import random

import torch
import torch.nn as nn
from torch.distributions import Categorical 

from nsfr.common import get_nsfr_model
from nsfr.utils.common import load_module

"""
Game State -> NSFR Logic Actor -> Logical Action Probabilites -> Environment Actions
                                                              -> Critic -> Value Estimation

1. Neural-Symbolic Actor: Uses the NSFR framework to derive actions through logical reasoning
2. Neural Network Critic: Estimates state values for reinforcement learning
3. Action Translation Layer: Maps logical predicates to environment actions
"""

class LogicalAgent(nn.Module):
    def __init__(self, env, rules, device, rng=None):
        super(LogicalAgent, self).__init__()
        self.device = device
        self.rng = random.Random() if rng is None else rng
        self.env = env

        """
        Initialization

        This creates:
                - An NSFR model as the logical policy (actor)
                - A neural network critic for value estimation
                - Action space mappings between logical predicates and environment actions
                - Exploration distribution for random action sampling 
        """
        # Create the NSFR logic policy
        self.logic_actor = get_nsfr_model(env.name, rules, device=device, train=True)
        self.prednames = self.logic_actor.get_prednames()

        # Create a simple critic to estimate value
        mlp_module_path = f"in/envs/{self.env.name}/mlp.py"
        module = load_module(mlp_module_path)
        self.critic = module.MLP(device=device, out_size=1, logic=True)

        # Action space setup
        self.num_actions = len(self.prednames)
        self.env_action_id_to_action_pred_indices = self._build_action_id_dict()
        
        # For exploration
        self.uniform = Categorical(
            torch.tensor([1.0 / self.num_actions for _ in range(self.num_actions)], device=device))
        self.upprior = Categorical(
            torch.tensor([0.9] + [0.1 / (self.num_actions-1) for _ in range(self.num_actions-1)], device=device))
    
    def forward(self):
        raise NotImplementedError

    def _print(self):
        print("==== Logic Policy ====")
        self.logic_actor.print_program()

    
    """
    Action Mapping

        This critical method:
            - Takes environment action names (e.g., "move_left")
            - Finds corresponding logical predicates in NSFR (e.g., "move_left", "move_left_away_from_alien")
            - Creates a mapping from environment actions to logical predicate indices
            - Adds dummy indices for actions with no matching predicate
    """
    def _build_action_id_dict(self):
        # Maps environment actions to predicate indices
        env_action_names = list(self.env.pred2action.keys())
        env_action_id_to_action_pred_indices = {}
        
        # Initialize dictionary
        for i, env_action_name in enumerate(env_action_names):
            env_action_id_to_action_pred_indices[i] = []
            
        # Fill dictionary
        for i, env_action_name in enumerate(env_action_names):
            exist_flag = False
            for j, action_pred_name in enumerate(self.logic_actor.get_prednames()):
                if env_action_name in action_pred_name:
                    env_action_id_to_action_pred_indices[i].append(j)
                    exist_flag = True
            if not exist_flag:
                # Add dummy index if no predicate matches this action
                dummy_index = len(self.logic_actor.get_prednames())
                env_action_id_to_action_pred_indices[i].append(dummy_index)
                
        return env_action_id_to_action_pred_indices
    

    """
    Action Distribution Conversion:
        - Takes raw predicate probabilites from the NSFR model
        - Converts to logits for numerical stability
        - For each environment action, finds all related logical predicates
        - Takes the maximum value (using torch.max) if multiple predicates map to one action
        - Converts back to probabilites using softmax
    """
    def to_action_distribution(self, raw_action_probs):
        # Convert logic policy outputs to action distribution
        batch_size = raw_action_probs.size(0)
        env_action_names = list(self.env.pred2action.keys())
        
        # Add dummy predicate for actions not covered by rules
        raw_action_probs = torch.cat([raw_action_probs, 
                                     torch.zeros(batch_size, 1, device=self.device)], dim=1)
        raw_action_logits = torch.logit(raw_action_probs, eps=0.01)
        
        # Collect action values
        dist_values = []
        for i in range(len(env_action_names)):
            if i in self.env_action_id_to_action_pred_indices:
                indices = torch.tensor(self.env_action_id_to_action_pred_indices[i], device=self.device)\
                    .expand(batch_size, -1)
                gathered = torch.gather(raw_action_logits, 1, indices)
                # Merge values (softor could be replaced with torch.max)
                merged = torch.max(gathered, dim=1)[0].unsqueeze(1)
                dist_values.append(merged)
        
        # Create action distribution
        action_values = torch.cat(dist_values, dim=1)
        action_dist = torch.softmax(action_values, dim=1)
        
        # Ensure right shape
        if action_dist.size(1) < self.env.n_raw_actions:
            zeros = torch.zeros(batch_size, 
                               self.env.n_raw_actions - action_dist.size(1), 
                               device=self.device, requires_grad=True)
            action_dist = torch.cat([action_dist, zeros], dim=1)
            
        return action_dist
    
    """
    Action Selection and Critic Evaluation
    
        This is the main interface for reinforcement learning algorithms:
            - Passes the game state through NSFR to get logical predicate probabilites
            - Translates these to environment action probabilites
            - Samples an action (fo exploration) or evaluates a provided action (for updates)
            - Computes necessary values for policy gradient algorithms
            - Returns everything needed for RL training
    """
    def get_action_and_value(self, logic_state, action=None):
        # Get raw action probabilities from NSFR
        raw_action_probs = self.logic_actor(logic_state)
        
        # Convert to environment action distribution
        action_dist = self.to_action_distribution(raw_action_probs)
        dist = Categorical(action_dist)
        
        # Sample action if not provided
        if action is None:
            action = dist.sample()
        
        # Get log probabilities and value
        logprob = dist.log_prob(action)
        value = self.critic(logic_state)
        
        return action, logprob, dist.entropy(), value
    
    def get_value(self, logic_state):
        return self.critic(logic_state)
