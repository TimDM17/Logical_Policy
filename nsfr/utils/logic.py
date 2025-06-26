from nsfr.fol.logic import *
from nsfr.fol.data_utils import DataUtils
from nsfr.fol.language import DataType
from nsfr.infer import InferModule
from nsfr.tensor_encoder import TensorEncoder



p_ = Predicate('.', 1, [DataType('spec')])
false = Atom(p_, [Const('__F__', dtype=DataType('spec'))])
true = Atom(p_, [Const('__T__', dtype=DataType('spec'))])


def get_lang(lark_path, lang_base_path, dataset):
    """
    Load the language of first-order logic from files.

    Read the language, clauses, background knowledge from files.
    Atoms are generated from the language

    1. Initializes the DataUtils object
    2. Loads the language definition (lang) with predicates and constants
    3. Loads clauses (rules) and background knowledge (facts)
    4. Calls generate_atoms(lang) to create all possible atoms from the language
    5. Returns all these components together as a complete logical system
    """
    du = DataUtils(lark_path=lark_path, lang_base_path=lang_base_path, dataset=dataset)
    lang = du.load_language() # This method build the vocabulary of the logical system by loading preds.txt, neural_preds.txt and const.txt
    clauses = du.get_clauses(lang) # This method loads logical clauses (rules) from a text file
    bk = du.get_bk(lang) # This method loads background knowledge (known facts) from a text file
    atoms = generate_atoms(lang)

    return lang, clauses, bk, atoms


def generate_atoms(lang):
    """
    This function systematically creates all possible ground atoms (atomic formulas with no variables)
    that can be expressed in the given logical language. It:
    
    1. Starts with special atoms false and true
    2. Iterates through each predicate in the language
    3. For each predicate:
        - Identifies required data types for its arguments
        - Collects all constants of those data types
        - Generates all possible combinations of arguments using itertools.product
        - Filter combinations to keep only valid ones
        - Creates Atom objects for each valid combination
    4. Returns the complete list of all possible atoms
    """
    spec_atoms = [false, true] # Start with special atoms
    atoms = []
    for pred in lang.preds:
        dtypes = pred.dtypes # Get data types for this predicate
        consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes] # Get constants of each type
        args_list = list(set(itertools.product(*consts_list))) # Generate all combinations
        # args_list = lang.get_args_by_pred(pred)
        args_str_list = []
        # args_mem = []
        for args in args_list:
            # Only include atoms where arguments are unique (or unary predicates)
            if len(args) == 1 or len(set(args)) == len(args):
                # if len(args) == 1 or (args[0] != args[1] and args[0].mode == args[1].mode):
                # if len(set(args)) == len(args):
                # if not (str(sorted([str(arg) for arg in args])) in args_str_list):
                atoms.append(Atom(pred, args))
                # args_str_list.append(
                #    str(sorted([str(arg) for arg in args])))
                # print('add atom: ', Atom(pred, args))
    return spec_atoms + sorted(atoms)


def build_infer_module(clauses, atoms, lang, device, m=3, infer_step=3, train=False):
    """
    Creates a differentiable forward-chaining inference module for neural-symbolic reasoning.

    This function builds the reasoning engine of the neural-symbolic system by:
    1. Encoding logical clauses into tensor representation via TensorEncoder
        - Converts symbolic rules into a 4D index tensor I [clauses, atoms, substitutions, body_length]
        - Handles variable substitutions and unification
    
    2. Creating an InferModule that performs differentiable logical inference
        - Applies rules repeatedly for 'infer_step' iterations
        - Supports trainable weights for rules when train=True
        - Uses soft logical operators to maintain gradient flow
    
    The resulting module takes a valuation tensor (from FactsConverter) and performs
    logical reasoning to derive higher-level facts. This is where rules like 
    "dangerous(X) :- close_by_alien(player, X)" are applied to derive facts like 
    "dangerous(alien1)" from "close_by_aliene(player, alien1)".

    Args:
        clauses: List of logical rules to apply
        atoms: List of all possible ground atoms
        lang: The logical language definition
        device: Computation device (CPU/GPU)
        m: Number of unique predicates in rule heads
        infer_step: Number of inference steps to perform
        train: Whether to use trainable weights

    Returns:
        im: Differentiable inference module
    """
    
    te = TensorEncoder(lang, atoms, clauses, device=device)
    # This creates a 4D index tensor "I" with shape [C, G, S, L]
    I = te.encode()
    # The InferModule uses the encoded tensor to perform differentiable logical inference
    # Performs forward-chaining inference through a fixed number of steps
    # Can have trainable weights for rule importance if train = True
    # Uses differentiable logical operations to ensure gradient flow
    im = InferModule(I, m=m, infer_step=infer_step, device=device, train=train)
    return im

"""
Finding Atom Indices:

This utility function serves a critical role in the NSFReasoner
1. It searches through the list of all possible atoms to find those with a specific predicate name
2. It returns the index of the first atom with that predicate name
3. This index is used to locate the corresponding probability in the valuation tensor

Applied in the NSFReasoner:
    The function specifically finds action predicates like "move_left" or "shoot" in the atom list,
    allowing the system to extract their probabilities from the valuation tensor.
    For example, if the predicate "mode_left" corresponds to atom index 42, then V_T[0, 42] gives
    the probability of that action being valid.

"""
def get_index_by_predname(pred_str, atoms):
    for i, atom in enumerate(atoms):
        if atom.pred.name == pred_str:
            return i
    assert 1, pred_str + ' not found.'