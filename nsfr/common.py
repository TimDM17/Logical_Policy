import os

from nsfr.facts_converter import FactsConverter
from nsfr.utils.logic import get_lang, build_infer_module
from nsfr.nsfr import NSFReasoner
from nsfr.valuation import ValuationModule

def get_nsfr_model(env_name: str, rules: str, device: str, train=False, explaine=False):
    
    # Determine file paths for accessing the logical components
    current_path = os.path.dirname(__file__) # Get the directory of the current module
    lark_path = os.path.join(current_path, 'lark/exp.lark') # Path to the Lark grammar file for parsing logical expressions
    lang_base_path = f"in/envs/{env_name}/logic/" # Environemnt-specific path to logical language definition files

    """
    Load the complete logical system for the environemnt:
    - lang: The language definition (predicates and constants)
    - clauses: The logical rules for reasoning
    - bk: Background knowledge (known facts)
    - atoms: All possible ground atoms generated from the language
    """
    lang, clauses, bk, atoms = get_lang(lark_path, lang_base_path, rules)

    """
    val_fn_path: 
    
    Define path to environment-specific valuation functions
    Each environment has its own valuation.py with predicates specific to that game's
    objects and mechanics. These functions calculate probabilistic truth values based
    on the game state.

    val_module:
    
    Create the valuation module that will translate atoms from game states to logical truth values
    The ValuationModule serves these specific technical purposes:
    
    1. Differentiability: By using PyTorch tensors and operations, enables gradient flow for learning
    2. Dynamic Reflection: Uses Python's introspection capabilities to dynamically load and bind functions
    3. Tensor Conversion: Implements a complex type system mapping between logical terms and tensor representations
    4. Batched Processing: Supports batch-wise operations for efficient GPU utilization
    """
    val_fn_path = f"in/envs/{env_name}/valuation.py"
    val_module = ValuationModule(val_fn_path, lang, device)

    """
    FactsConverter: Creates complete valuation vector for all atoms
                    This is the complete "world state" in logical terms, representing which
                    logical facts are true in the current game state
    """
    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device)

    """
    
    """
    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)
    m = len(prednames)
    # m = 5
    IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device)