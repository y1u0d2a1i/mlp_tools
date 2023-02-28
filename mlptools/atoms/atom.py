import numpy as np

from ase import Atoms
from ase.neighborlist import NeighborList

from ovito.modifiers import CoordinationAnalysisModifier
from ovito.pipeline import StaticSource, Pipeline
from ovito.io.ase import ase_to_ovito

class MLPAtoms:
    def __init__(self, cell, coord, energy, force,  n_atoms, structure_id=None, symbols=None, frame=None, additional_info=None) -> None:
        self.cell = cell
        self.coord = coord
        self.energy = energy
        self.force = force
        self.n_atoms = n_atoms
        self.structure_id = structure_id
        self.symbols = symbols
        self.frame = frame
        self.additional_info = additional_info

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

    def get_ase_atoms(self):
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
        aseatoms = self.get_ase_atoms()
        min_distance_list = [min(distance[np.where(distance != 0.0)]) for distance in aseatoms.get_all_distances(mic=True)]
        # nl = NeighborList([rcut/2]*self.n_atoms, self_interaction=False, bothways=False)
        # nl.update(aseatoms)

        # min_distance_list = []
        # for i in range(self.n_atoms): 
        #     nearest_neighbors = nl.get_neighbors(i)[0]
        #     distances = aseatoms.get_distances(i, nearest_neighbors, mic=True)
        #     print(distances)
        #     min_distance_list.append(min(distances)))
        
        if len(min_distance_list) > 0:
            self.distance_btw_nearest_neighbor = min(min_distance_list)
        else:
            print(f"There's no neighbors")

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