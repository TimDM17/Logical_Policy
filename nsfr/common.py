import os

from nsfr.facts_converter import FactsConverter
from nsfr.utils.logic import get_lang, build_infer_module
from nsfr.nsfr import NSFReasoner
from nsfr.valuation import ValuationModule


"""
Game State Tensor (Z)
      ↓
FactsConverter (FC)
      ↓
Initial Valuation Vector (V_0)
      ↓
InferModule (IM)
      ↓
Final Valuation Vector (V_T)
      ↓
Action Probability Extraction
"""

def get_nsfr_model(env_name: str, rules: str, device: str, train=False, explain=False):
    
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
    
    Create the valuation module that will translate a object-centric representation to a logical truth values
    Evaluates basic predicates from game state -> "Player is near alien"

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
                    Creates valuation vector for all atoms -> "World state"
    """
    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device)

    """
    Creating the inferece module - the component responsible for applying logical rules
    to derive new facts

    build_infer_module: Applies logical rules to derive higher-level facts -> "Player is in danger"

    This is where the "symbolic reasoning" part of neural-symbolic integration happens.
    While the earlier components ground predicates in the game state, this module applies the logical rules
    that encode game strategy or policy. It's the component that takes basic facts like
    "player near alien" and derives conclusions like "player should move right".
    """
    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)
    m = len(prednames)
    # m = 5
    IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device)


    """
    Neuro-Symbolic Forward Reasoner

    The NSFReasoner class is the central orchestrator
    that integrates all components of the neural-symbolic reasoning system.
    It provides a complete differentiable pipeline from raw game state observations to logically-
    derived action probabilities

    1. Differentiable Logic: Implements a fully differentiable logical reasoning system
       that can be trained with gradient-based methods
    2. Perception-Logic Integration: Bridges the gap between neural perception and symbolic reasoning
    3. Zero-Shot Reasoning: Can apply logical rules to situations never seen before
    4. Explainability: Provides transparency into the reasoning process, unlike black-box neural networks
    """
    NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk,
                        clauses=clauses, device=device, train=train, explain=explain)
    return NSFR




