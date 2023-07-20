from ase import Atoms
from ase.build import make_supercell
from typing import List

def generate_crystal(
    atoms:Atoms, 
    displacement:float, 
    scale:float, 
    supercell_mtx: List[List[int]]=None) -> Atoms:
    """generate crystal by adding small displacement while compressing and expanding crystal

    Parameters
    ----------
    atoms : Atoms
        initial ase atoms
    displacement : float
        how much to displace the atom between 0 and 1
    scale : float
        how much to scale the cell

    Returns
    -------
    Atoms
        modified ase atoms
    """
    atoms = atoms.copy()
    if displacement < 0 or displacement > 1:
        raise Exception('displacement should be between 0 and 1')
    
    if supercell_mtx is not None:
        atoms = make_supercell(atoms, supercell_mtx)
    atoms.rattle(stdev=displacement)
    atoms.set_cell(atoms.get_cell() * scale, scale_atoms=True)
    return atoms