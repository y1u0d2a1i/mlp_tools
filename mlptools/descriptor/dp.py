import numpy as np
from deepmd.infer import DeepPot
from typing import Tuple


from mlptools.atoms.atom import MLPAtoms


def get_descriptor_vector(atoms: MLPAtoms, path2model:str) -> np.ndarray:
    """get D vector from trained dp model for a given structure

    Args:
        atoms (MLPAtoms): _
        path2model (str): path to dp model

    Returns:
        np.ndarray: (1, n_atoms, dimension of D)
    """
    dp = DeepPot(path2model)
    coord = atoms.coord.reshape([1, -1])
    cell = atoms.cell.reshape([1, -1])
    atype = np.zeros(atoms.n_atoms)
    return dp.eval_descriptor(coord, cell, atype)


def get_predictions(atoms: MLPAtoms, path2model:str) -> Tuple:
    """get energy, force and virial from trained dp model for a given structure

    Args:
        atoms (MLPAtoms): _description_
        path2model (str): path to dp model

    Returns:
        Tuple: (energy, force, virial)
    """
    dp = DeepPot(path2model)
    coord = atoms.coord.reshape([1, -1])
    cell = atoms.cell.reshape([1, -1])
    atype = np.zeros(atoms.n_atoms)
    energy, force, virial = dp.eval(coord, cell, atype)
    return (energy, force, virial)