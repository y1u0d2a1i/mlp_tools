import numpy as np

from ase import Atoms
from ase.neighborlist import NeighborList

from ovito.modifiers import CoordinationAnalysisModifier
from ovito.pipeline import StaticSource, Pipeline
from ovito.io.ase import ase_to_ovito
from typing import Dict

from mlptools.analyzer.nearest_neighbor import NearestNeighborCalculator

class MLPAtoms:
    def __init__(self, cell, coord, energy, force, n_atoms, total_magnetization=None, structure_id=None, symbols=None, frame=None, additional_info=None, path=None, ase_atoms=None) -> None:
        self.cell = cell
        self.coord = coord
        self.energy = energy
        self.force = force
        self.total_magnetization = total_magnetization
        self.n_atoms = n_atoms
        self.structure_id = structure_id
        self.symbols = symbols
        self.frame = frame
        self.additional_info = additional_info
        self.path = path
        self.set_ase_atoms()

        self.distance_btw_nearest_neighbor = None
    
    def get_volume(self):
        cell = self.cell
        a = cell[0]
        b = cell[1]
        c = cell[2]
        vol = np.dot(a, np.cross(b,c))
        return round(vol, 3)

    def get_atomic_distance(self):
        if self.n_atoms != 2:
            raise Exception('This function is only available for dimer data')
        return np.linalg.norm(self.coord[0] - self.coord[1])

    def get_atomic_energy(self):
        return self.energy / self.n_atoms

    def get_atomic_volume(self):
        return self.get_volume() / self.n_atoms
    
    def set_ase_atoms(self):
        self.ase_atoms = Atoms(
            symbols=self.symbols,
            positions=self.coord,
            cell=self.cell,
            pbc=(1,1,1)
            )

    def get_ase_atoms(self):
        if self.ase_atoms is not None:
            return self.ase_atoms
        
        if self.symbols is None:
            raise Exception('Set symbols')
        return Atoms(
            symbols=self.symbols,
            positions=self.coord,
            cell=self.cell,
            pbc=(1,1,1)
            )

    def set_distance_btw_nearest_neighbor(self) -> None:
        """set distance(ang) between nearest neighbor

        """
        calculator = NearestNeighborCalculator(self.ase_atoms)
        self.distance_btw_nearest_neighbor = calculator.get_nearest_neighbor_dict()

    def get_rdf(self, rcut=6, bins=100) -> np.ndarray:
        """get radial distribution function val

        Args:
            rcut (int, optional): cutoff radius. Defaults to 6.
            bins (int, optional): _description_. Defaults to 100.

        Returns:
            np.ndarray: 1st: distance, 2nd: rdf value
        """
        ase_atoms = self.get_ase_atoms()
        pipeline = Pipeline(source = StaticSource(data=ase_to_ovito(ase_atoms)))
        modifier = CoordinationAnalysisModifier(cutoff=rcut, number_of_bins=bins, partial=True)
        pipeline.modifiers.append(modifier)
        rdf_table = pipeline.compute().tables['coordination-rdf']
        return rdf_table.xy()
    
    def get_rdf_for_multiple_species(self, rcut=6, bins=100) -> Dict[str, np.ndarray]:
        """get radial distribution function table

        Args:
            rcut (int, optional): cutoff radius. Defaults to 6.
            bins (int, optional): _description_. Defaults to 100.

        Returns:
            Dict[str, np.ndarray]: distance and rdf value for each species combination
        """
        ase_atoms = self.get_ase_atoms()
        pipeline = Pipeline(source = StaticSource(data=ase_to_ovito(ase_atoms)))
        modifier = CoordinationAnalysisModifier(cutoff=rcut, number_of_bins=bins, partial=True)
        pipeline.modifiers.append(modifier)
        rdf_table = pipeline.compute().tables['coordination-rdf']
        rdf_component_name = rdf_table.y.component_names
        rdf_dict = {}
        for i, name in enumerate(rdf_component_name):
            if not "distance" in rdf_dict.keys():
                rdf_dict["distance"] = rdf_table.xy()[:, 0]
            rdf_dict[name] = rdf_table.y[:, i]
        return rdf_dict