import os
import numpy as np

from abc import ABC, abstractmethod
from ase.io import read

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


# Quantum espresso
class PWscfParser(BaseParser):
    def __init__(self, path_to_target, name_scf_in='scf.in', name_scf_out='scf.out') -> None:
        super().__init__()
        self.path_to_target = path_to_target
        self.name_scf_in = name_scf_in
        with open(f'{path_to_target}/{name_scf_in}') as f:
            self.I_lines = [s.strip() for s in f.readlines()]
        with open(f'{path_to_target}/{name_scf_out}') as f:
            self.O_lines = [s.strip() for s in f.readlines()]
        
        self.validate_o_lines()
        
        num_atom = self.I_lines[get_param_idx('nat', self.I_lines)]
        num_atom = num_atom.split(' ')[-1]
        self.num_atom = int(num_atom)
        
        cell_idx = get_param_idx('CELL_PARAMETERS', self.I_lines)
        self.cell = self.I_lines[cell_idx+1 : cell_idx+4]
        
        coord_idx = get_param_idx('ATOMIC_POSITIONS', self.I_lines)
        self.coord = self.I_lines[coord_idx+1 : coord_idx+1+self.num_atom]
        
        self.au2ang = self.get_au2ang()
        self.rv2ev = 13.60

        self.structure_id = list(filter(lambda x: 'mp-' in x, path_to_target.split('/')))[0]

        self.qeltype = 'SCF'

    
    def get_cell(self):
        cell_matlix = self.cell.copy()
        cell_matlix = np.array([list(map(lambda x: float(x), remove_empty_from_array(l.split(' ')))) for l in cell_matlix])
        return cell_matlix
    

    def get_coord(self):
        structure = read(os.path.join(self.path_to_target, self.name_scf_in), format='espresso-in')
        return structure.get_positions()


    def get_energy(self, is_ev=True):
        energy_idx = get_param_idx('!', self.O_lines)
        energy = float(list(filter(lambda l: l != '', self.O_lines[energy_idx].split(' ')))[-2])
        if is_ev:
            return energy * self.rv2ev
        else:
            return energy
        
            
    def get_force(self, is_ev_ang=True):
        force_idx = get_param_idx('Forces acting on atoms (cartesian axes, Ry/au):', self.O_lines)  
        start = force_idx+2
        end = force_idx+2 + self.num_atom
        forces = [list(filter(lambda l: l != '', line.split(' ')))[-3:] for line in self.O_lines[start:end]]
        if is_ev_ang:
            forces = [ np.array([float(i) for i in force]) * (self.rv2ev / self.au2ang) for force in forces]
        return np.array(forces)
    

    def get_structure_id(self):
        return self.structure_id


    def get_n_atoms(self):
        return self.num_atom
    
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