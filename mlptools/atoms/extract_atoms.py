from ase import Atoms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List

def extract_atoms(
        atoms:Atoms, 
        x_range:List[float|int], 
        y_range:List[float|int], 
        z_range:List[float|int]) -> Atoms:
    """Extract atoms from a larger cell based on the given ranges

    Args:
        atoms (Atoms): ase atoms object
        x_range (List[float | int]): _description_
        y_range (List[float | int]): _description_
        z_range (List[float | int]): _description_

    Returns:
        Atoms: extracted atoms object
    """    
    # check if x_range, y_range, z_range are valid
    if x_range[0] > x_range[1]:
        raise ValueError("x_range[0] must be smaller than x_range[1]")
    if y_range[0] > y_range[1]:
        raise ValueError("y_range[0] must be smaller than y_range[1]")
    if z_range[0] > z_range[1]:
        raise ValueError("z_range[0] must be smaller than z_range[1]")
    if x_range[0] < 0 or x_range[1] > atoms.cell[0, 0]:
        raise ValueError("x_range must be within the cell range")
    if y_range[0] < 0 or y_range[1] > atoms.cell[1, 1]:
        raise ValueError("y_range must be within the cell range")
    if z_range[0] < 0 or z_range[1] > atoms.cell[2, 2]:
        raise ValueError("z_range must be within the cell range")

    extracted_atoms = atoms[
        (atoms.positions[:, 0] > x_range[0]) & (atoms.positions[:, 0] < x_range[1]) &
        (atoms.positions[:, 1] > y_range[0]) & (atoms.positions[:, 1] < y_range[1]) &
        (atoms.positions[:, 2] > z_range[0]) & (atoms.positions[:, 2] < z_range[1])
    ]
    extracted_atoms.set_positions(extracted_atoms.positions - np.array([x_range[0], y_range[0], z_range[0]]))
    extracted_atoms.set_cell([
        [x_range[1] - x_range[0], 0, 0],
        [0, y_range[1] - y_range[0], 0],
        [0, 0, z_range[1] - z_range[0]]
    ])
    extracted_atoms.wrap()

    if len(extracted_atoms) == 0:
        raise ValueError("No atoms were extracted")

    return extracted_atoms