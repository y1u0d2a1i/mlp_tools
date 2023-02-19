import numpy as np
from deepmd.infer import DeepPot
from typing import Tuple


from mlptools.atoms.atom import MLPAtoms


def get_descriptor_vector(atoms: MLPAtoms, path2model:str=None, model:DeepPot=None) -> np.ndarray:
    """get D vector from trained dp model for a given structure

    Args:
        atoms (MLPAtoms): _
        path2model (str): path to dp model
        model (DeepPot): trained model object

    Returns:
        np.ndarray: (1, n_atoms, dimension of D)
    """
    if path2model is None and model is None:
        raise Exception('set either path2model or model object')
    
    if model is None:
        model = DeepPot(path2model)
    coord = atoms.coord.reshape([1, -1])
    cell = atoms.cell.reshape([1, -1])
    atype = np.zeros(atoms.n_atoms)
    return model.eval_descriptor(coord, cell, atype)


def get_predictions(atoms: MLPAtoms, path2model:str=None, model:DeepPot=None) -> Tuple:
    """get energy, force and virial from trained dp model for a given structure

    Args:
        atoms (MLPAtoms): _description_
        path2model (str): path to dp model
        model (DeepPot): trained model object
        
    Returns:
        Tuple: (energy, force, virial)
    """
    if path2model is None and model is None:
        raise Exception('set either path2model or model object')
    
    if model is None:
        model = DeepPot(path2model)
    dp = DeepPot(path2model)
    coord = atoms.coord.reshape([1, -1])
    cell = atoms.cell.reshape([1, -1])
    atype = np.zeros(atoms.n_atoms)
    energy, force, virial = dp.eval(coord, cell, atype)
    return (energy, force, virial)