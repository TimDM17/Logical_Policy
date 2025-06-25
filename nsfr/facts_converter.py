import torch
import torch.nn as nn
from .fol.logic import NeuralPredicate
from tqdm import tqdm


class FactsConverter(nn.Module):
    """
    FactsConverter converts the output from the perception module to the valuation vector.

    FactsConverter transforms raw game state tensors into a complete valuation vector.

    This module:
    1. Takes object-centric representations from perception/observation
    2. Evaluates all neural predicates using the ValuationModule
    3. Incorporates background knowledge (known facts)
    4. Produces a valuation vector where each element represents an atom's probability

    This creates the "ground truth" tensor that the logical reasoning system uses
    to evaluate rules and make decisions.

    """

    def __init__(self, lang, valuation_module, device=None):
        """
        Initialize the FactsConverter.

        Args:
            lang: Logical language definition with predicates and constants
            valuation_module: Module for evaluating predicates
            device: Computation device (CPU/GPU)
        """
        super(FactsConverter, self).__init__()
        # self.e = perception_module.e
        self.e = 0 # Number of entities
        #self.d = perception_module.d
        self.d = 0 # Feature dimension
        self.lang = lang # The logical language (predicates, constants)
        self.vm = valuation_module  # valuation functions / ValuationModule for evaluating neural predicates
        self.device = device # Computation device (CPU/GPU)

    def __str__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.d)

    def __repr__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.d)

    def forward(self, Z, G, B):
        """
        Forward pass converts game state to valuation vector.

        This is an alias for the convert method to follow PyTorch conventions
        """
        return self.convert(Z, G, B)

    def get_params(self):
        return self.vm.get_params()

    def init_valuation(self, n, batch_size):
        """
        Initialize a valuation vector with 'true' set to 1.0 and others to 0.0.

        Args:
            n: Number of atoms in the language
            batch_size: Batch dimension size

        Returns:
            Initialized valuation tensor [batch_size, n]
        """
        v = torch.zeros((batch_size, n)).to(self.device)
        v[:, 1] = 1.0
        return v

    def filter_by_datatype(self):
        pass

    def to_vec(self, term, zs):
        pass

    def __convert(self, Z, G):
        # Z: batched output
        vs = []
        for zs in tqdm(Z):
            vs.append(self.convert_i(zs, G))
        return torch.stack(vs)

    # Core Conversion Process
    def convert(self, Z, G, B):
        """
        Convert game state into logical valuations for all atoms

        This method creates a complete valuation vector by:
        1. Setting neural predicates based on valuation functions (calculated from game state)
        2. Setting background knowledge atoms to true (1.0)
        3. Ensuring special atoms (true/false) have correct values

        Args:
            Z: Game state tensor [batch_size, num_objects, features]
            G: List of all possible ground atoms in the language
            B: Background knowledge (atoms that are known to be true)

        Returns:
            V: Valuation tensor [batch_size, num_atoms] with probabilities for each atom 
               This vector is the complete "world state" in logical terms, representing which
               logical facts are true in the current game state   
        """
        batch_size = Z.size(0)

        # V = self.init_valuation(len(G), Z.size(0))
        # Initialize valuation vector with zeroes
        V = torch.zeros((batch_size, len(G))).to(
            torch.float32).to(self.device)
        # Process each ground atom
        for i, atom in enumerate(G):
            if type(atom.pred) == NeuralPredicate and i > 1:
                # For neural predicates, compute probability using valuation module
                V[:, i] = self.vm(Z, atom)
            elif atom in B:
                # V[:, i] += 1.0
                # For atoms in background knowledge, set to 1.0 (true)
                V[:, i] += torch.ones((batch_size,)).to(
                    torch.float32).to(self.device)
        # Always set the second atom (true) to 1.0
        V[:, 1] = torch.ones((batch_size,)).to(
            torch.float32).to(self.device)
        return V

    def convert_i(self, zs, G):
        v = self.init_valuation(len(G))
        for i, atom in enumerate(G):
            if type(atom.pred) == NeuralPredicate and i > 1:
                v[i] = self.vm.eval(atom, zs)
        return v

    def call(self, pred):
        return pred
