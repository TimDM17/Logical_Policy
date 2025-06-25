import torch
import sys
import importlib.util


def bool_to_probs(bool_tensor: torch.Tensor):
    """Converts the values of a tensor from Boolean to probability values by
     slightly 'smoothing' them (1 to 0.99 and 0 to 0.01)."""
    return torch.where(bool_tensor, 0.99, 0.01)


def load_module(path: str):
    """
    Dynamically loads a Python module from a file path.

    This functions allows importing module that aren't in the Python path
    or whose location is only known at runtime. It's used to load environment-specific
    code like valuation function for different game environment

    Args:
        path: File path to the Python module to be loaded

    Returns:
        The loaded module object with all its functions and classes available
    """
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)
    return module