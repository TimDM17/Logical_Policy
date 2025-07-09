from typing import Dict
from abc import ABC
from nsfr.utils.common import load_module
import torch


"""
The VectorizedNudgeBaseEnv class serves as a critical abstraction layer that bridges the gap between
raw game environments and the neural-symbolic reasoning system. It's designed as
an abstract base class (ABC) that provides:
    
    - Standardized Environment Interface: Creates a consistent API regardless of the underlying game environment
    - Multiple State Representations: Maintains parallel representations for different processing needs:
        - Logic state: Object-centric tensor representation for NSFR logical reasoning
        - Neural state: Observation representation for neural network processing
    - Action Space Translation: Handles the complex mapping between logical predicates and environment action indices
    - Dual Operation Modes: Supports both standard PPO reinforcement learning and logical agent modes
"""
class VectorizedNudgeBaseEnv(ABC):
    name: str
    pred2action: Dict[str, int]  # predicate name to action index
    env: object  # the wrapped environment
    raw_env: object # the raw RGB environment, not RAM

    def __init__(self, mode: str):
        self.mode = mode  # either 'ppo' or 'logic'

    """
    Environment Lifecycle Management:
        These methods standardize the environment interaction cycle while handling the necessary conversions.
    """
    def reset(self) -> tuple[torch.tensor, torch.tensor]:
        """Returns logic_state, neural_state"""
        raise NotImplementedError

    def step(self, action, is_mapped: bool = False) -> tuple[tuple[torch.tensor, torch.tensor], float, bool]:
        """If is_mapped is False, the action will be mapped from model action space to env action space.
        I.e., if is_mapped is True, this method feeds 'action' into the wrapped env directly.
        Returns (logic_state, neural_state), reward, done"""
        raise NotImplementedError

    """
    State Representation Conversion:
        These three methods handle the critical transformation from raw game states to the structured representations
        needed by different components of the system
    """
    def extract_logic_state(self, raw_state) -> torch.tensor:
        """Turns the raw state representation into logic representation."""
        raise NotImplementedError

    def extract_neural_state(self, raw_state) -> torch.tensor:
        """Turns the raw state representation into neural representation."""
        raise NotImplementedError

    def convert_state(self, state, raw_state) -> tuple[torch.tensor, torch.tensor]:
        return self.extract_logic_state(state), self.extract_neural_state(raw_state)

    """
    Bidirectional Action Mapping:
        This handles the bidirectional translation between:
            - Logical action predicates from the reasoning system (move_left)
            - Numeric action indices expected by the game environment (e.g, 0, 1, 2)
    """
    def map_action(self, model_action) -> int:
        """Converts a model action to the corresponding env action."""
        if self.mode == 'ppo':
            return model_action + 1
        else:  # logic
            pred_names = list(self.pred2action.keys())
            for pred_name in pred_names:
                if pred_name in model_action:
                    return self.pred2action[pred_name]
            raise ValueError(f"Invalid predicate '{model_action}' provided. "
                             f"Must contain any of {pred_names}.")

    def n_actions(self) -> int:
        # return len(set(self.pred2action.values())) -> Fix: Count unique action values
        return len(list(set(self.pred2action.items()))) # bug: item() returns (key,values) tuples

    """
    Environment Factory:
        This factory method dynamically loads environment-specific implementations based on name
    """
    @staticmethod
    def from_name(name: str, **kwargs):
        env_path = f"in/envs/{name}/env_vectorized.py"
        env_module = load_module(env_path)
        return env_module.VectorizedNudgeEnv(**kwargs)

    def close(self):
        pass
