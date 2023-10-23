from pydantic import BaseModel
from ase import Atoms
from typing import List, Dict
from itertools import combinations_with_replacement



class Bond(BaseModel):
    first_atomic_symbol: str
    second_atomic_symbol: str

    @property
    def bond_str(self) -> str:
        return f"{self.first_atomic_symbol}-{self.second_atomic_symbol}"



class NearestNeighborCalculator():
    def __init__(self, ase_atoms:Atoms) -> None:
        if isinstance(ase_atoms, Atoms):
            self.ase_atoms = ase_atoms
        else:
            raise Exception('ase_atoms must be instance of ase.Atoms')
    

    def get_nearest_neighbor(self, target_bond:Bond) -> float:
        """get nearest neighbor distance for target-bond from distance matrix generated from ase.Atoms

        Parameters
        ----------
        target_bond : Bond


        Returns
        -------
        float
            nearest neighbor distance for target-bond
        """
        distance_matrix = self.ase_atoms.get_all_distances(mic=True)
        chemical_symbols = self.ase_atoms.get_chemical_symbols()

        distance_list = []
        for i in range(len(chemical_symbols)):
            if chemical_symbols[i] != target_bond.first_atomic_symbol:
                continue
            for j in range(len(chemical_symbols)):
                if chemical_symbols[j] != target_bond.second_atomic_symbol:
                    continue
                if i == j:
                    continue
                print(chemical_symbols[i], chemical_symbols[j],distance_matrix[i, j])
                distance_list.append(distance_matrix[i, j])

        print(f"Nearst neighbor distance for {target_bond} is {min(distance_list)}")
        return min(distance_list)
    

    def get_possible_bonds(self, chemical_symbols: List[str]) -> List[Bond]:
        """get possible bonds from chemical_symbols

        Parameters
        ----------
        chemical_symbols : List[str]
            list of chemical symbols

        Returns
        -------
        List[Bond]
            list of possible bonds
        """
        unique_chemical_symbol_list = list(set(chemical_symbols))
        possible_bonds = list(combinations_with_replacement(unique_chemical_symbol_list, 2))
        print(possible_bonds)

        return [Bond(first_atomic_symbol=bond[0], second_atomic_symbol=bond[1]) for bond in possible_bonds]


    def get_nearest_neighbor_dict(self) -> Dict[str, float]:
        """get nearest neighbor distance for all possible bonds

        Returns
        -------
        Dict[str, float]
            dictionary of nearest neighbor distance for all possible bonds
            key is bond_str, value is nearest neighbor distance
        """
        chemical_symbols = self.ase_atoms.get_chemical_symbols()
        all_bonds = self.get_possible_bonds(chemical_symbols)
        return {bond.bond_str: self.get_nearest_neighbor(bond) for bond in all_bonds}
