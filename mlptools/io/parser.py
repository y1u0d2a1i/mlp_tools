import os
import numpy as np

from abc import ABC, abstractmethod
from ase.io import read
from ase.io.espresso import read_espresso_out, read_espresso_in


from mlptools.utils.utils import get_param_idx, remove_empty_from_array

class BaseParser(ABC):
    @abstractmethod
    def get_cell(self):
        raise NotImplementedError()

    @abstractmethod
    def get_coord(self):
        raise NotImplementedError()

    @abstractmethod
    def get_energy(self):
        raise NotImplementedError()

    @abstractmethod
    def get_force(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_n_atoms(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_structure_id(self):
        raise NotImplementedError()

    @abstractmethod
    def get_total_magnetization(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_ase_atoms(self):
        raise NotImplementedError()


# Quantum espresso
class PWscfParser(BaseParser):
    def __init__(self, path_to_target, name_scf_in='scf.in', name_scf_out='scf.out', structure_id=None) -> None:
        super().__init__()
        self.path_to_target = path_to_target
        self.name_scf_in = name_scf_in

        with open(os.path.join(path_to_target, name_scf_out)) as f:
            atom_gen = read_espresso_out(f, index=slice(None))
            ase_atoms = next(atom_gen)
            print(ase_atoms)
        self.ase_atoms = ase_atoms

        with open(f'{path_to_target}/{name_scf_in}') as f:
            self.I_lines = [s.strip() for s in f.readlines()]
        with open(f'{path_to_target}/{name_scf_out}') as f:
            self.O_lines = [s.strip() for s in f.readlines()]
        
        self.validate_o_lines()
        
        self.num_atom = ase_atoms.get_global_number_of_atoms()
        self.cell = np.array([vec for vec in ase_atoms.cell])
        self.coord = ase_atoms.positions

        mpid = list(filter(lambda x: 'mp-' in x, path_to_target.split('/')))
        if structure_id is None and len(mpid) > 0:
            self.structure_id = mpid[0]
        else:
            self.structure_id = structure_id

        self.qeltype = 'SCF'

    
    def get_cell(self):
        return self.cell
    

    def get_coord(self):
        return self.coord


    def get_energy(self, is_ev=True):
        return self.ase_atoms.get_potential_energy()
        
            
    def get_force(self, is_ev_ang=True):
        return self.ase_atoms.get_forces()
    

    def get_structure_id(self):
        return self.structure_id


    def get_n_atoms(self):
        return self.num_atom
    
    def get_ase_atoms(self):
        return self.ase_atoms
    
    def get_total_magnetization(self):
        total_mag = list(filter(lambda x: 'total magnetization' in x, self.O_lines))
        if len(total_mag) == 0:
            return None
        
        mag_val_idx = 3
        final_mag = float(list(filter(lambda x: x != '', total_mag[-1].split(' ')))[mag_val_idx])
        return final_mag


    def validate_o_lines(self):
        if 'JOB DONE.' not in self.O_lines:
            raise Exception("invalid: job didnot finished")
            
        for line in self.O_lines:
            if 'convergence NOT achieved' in line:
                raise Exception('invalid: convergence NOT achieved')

            if 'SCF correction compared to forces is large' in line:
                raise Exception('invalid: Unreliable scf result')

    
    def get_au2ang(self):
        """ Get convert factor from au(Atomic unit) to angstrom see link below.
        https://courses.engr.illinois.edu/mse404ela/sp2019/6.DFT-walkthrough.html

        Returns:
            float: factor
        """
        # lattice_constant = float(self.cell[0].split(' ')[0])
        # au2ang = lattice_constant
        # return au2ang
        au2ang = 0.529177211
        return au2ang