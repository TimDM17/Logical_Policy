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
