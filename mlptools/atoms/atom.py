import numpy as np

from ase import Atoms
from ovito.modifiers import CoordinationAnalysisModifier
from ovito.pipeline import StaticSource, Pipeline
from ovito.io.ase import ase_to_ovito

class MLPAtoms:
    def __init__(self, cell, coord, energy, force,  n_atoms, structure_id=None, symbols=None, frame=None) -> None:
        self.cell = cell
        self.coord = coord
        self.energy = energy
        self.force = force
        self.n_atoms = n_atoms
        self.structure_id = structure_id
        self.symbols = symbols
        self.frame = frame
    
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

    def get_ase_atoms(self):
        if self.symbols is None:
            raise Exception('Set symbols')
        return Atoms(
            symbols=self.symbols,
            positions=self.coord,
            cell=self.cell,
            pbc=(1,1,1)
            )

    def get_rdf(self, rcut, bins):
        ase_atoms = self.get_ase_atoms()
        pipeline = Pipeline(source = StaticSource(data=ase_to_ovito(ase_atoms)))
        modifier = CoordinationAnalysisModifier(cutoff=rcut, number_of_bins=bins, partial=True)
        pipeline.modifiers.append(modifier)
        rdf_table = pipeline.compute().tables['coordination-rdf']
        return rdf_table.xy()