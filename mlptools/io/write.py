from mlptools.atoms.atom import MLPAtoms
from mlptools.utils.utils import flatten
from typing import List
from abc import ABC, abstractmethod
from ase import Atoms
import os
import numpy as np  


def write_from_atoms(atoms: MLPAtoms, format: str, structure_id=None) -> List[str]:
    if format == 'n2p2':
        writer = N2p2Writer(atoms, structure_id=structure_id)
    else:
        raise Exception('Not supported format')
    
    return writer.output()

class BaseWriter(ABC):
    def __init__(self, atoms) -> None:
        self.atoms = atoms

    @abstractmethod
    def output(self):
        raise NotImplementedError

class QuantumEspressoWriter(BaseWriter):
    def __init__(self, atoms: Atoms, path2template: str, scf_filename="scf.in", out_dir=None) -> None:
        super().__init__(atoms)
        self.template = path2template
        self.scf_filename = scf_filename
        self.out_dir = out_dir
    
    def read_template(self):
        # read scf.in.template
        with open(os.path.join(self.template, self.scf_filename), 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        return lines
    
    def get_param_idx(self, param, lines):
        for i, line in enumerate(lines):
            if param in line:
                return i
        return None
    
    def flatten(self, lst):
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(flatten(item))
            else:
                result.append(item)
        return result
    
    def output(self):
        scf_input_lines = self.read_template()
        # change outdir
        if self.out_dir is not None:
            outdir_idx = self.get_param_idx('outdir', scf_input_lines)
            scf_input_lines[outdir_idx] = f"outdir = '{self.out_dir}'"
        # change num of atoms
        num_atoms = self.atoms.get_global_number_of_atoms()
        num_atoms_idx = self.get_param_idx('nat', scf_input_lines)
        scf_input_lines[num_atoms_idx] = f'nat = {num_atoms}'

        cell_lines = []
        for vec in self.atoms.get_cell():
            cell_line = ' '.join(map(str, vec))
            cell_lines.append(cell_line)
            # print(cell_line)
        # change cell
        cell_idx = self.get_param_idx('CELL_PARAMETERS {angstrom}', scf_input_lines) 
        # insert list to list
        scf_input_lines.insert(cell_idx+1, cell_lines[0])
        scf_input_lines.insert(cell_idx+2, cell_lines[1])
        scf_input_lines.insert(cell_idx+3, cell_lines[2])

        position_lines = []
        for symbol, scaled_position in zip(self.atoms.get_chemical_symbols(), self.atoms.get_scaled_positions()):
            position_line = f'{symbol} ' + ' '.join(map(str, scaled_position))
            position_lines.append(position_line)
            # print(position_line)
        # change position
        position_idx = self.get_param_idx('ATOMIC_POSITIONS {crystal}', scf_input_lines)
        scf_input_lines.insert(position_idx+1, position_lines)

        scf_input_lines = self.flatten(scf_input_lines)
        return scf_input_lines

class N2p2Writer(BaseWriter):
    def __init__(
            self, 
            atoms: MLPAtoms, 
            is_comment=True, 
            structure_id=None,
            has_calculator=True
        ) -> None:
        self.is_comment = is_comment
        self.atoms = atoms
        self.structure_id = structure_id
        self.has_calculator = has_calculator
        

    def n2p2_comment(self):
        return f'comment {self.structure_id} .'

    def n2p2_cell(self):
        line = []
        cell = self.atoms.cell
        for l_vec in cell:
            l_vec = [str(i) for i in list(l_vec)]
            tmp = ' '.join(l_vec)
            line.append(f'lattice {tmp}')
        return line
    
    def n2p2_atom(self):
        line = []
        coord = self.atoms.coord
        force = self.n2p2_force()
        species = self.atoms.ase_atoms.get_chemical_symbols()
        for c, f, specie in zip(coord, force, species):
            c = [str(i) for i in list(c)]
            f = [str(i) for i in list(f)]
            tmp_c = ' '.join(c)
            tmp_f = ' '.join(f)
            line.append(f'atom {tmp_c} {specie} 0 0 {tmp_f}')
        return line
    
    def n2p2_energy(self):
        if self.has_calculator:
            energy = self.atoms.energy
        else:
            energy = 0.0
        return f'energy {energy}'


    def n2p2_force(self):
        if self.has_calculator:
            return self.atoms.force
        else:
            return np.zeros((len(self.atoms), 3))
    
    def n2p2_charge(self):
        return 'charge 0.0'
    
    def output(self):
        if self.is_comment:
            block = [
                'begin',
                self.n2p2_comment(),
                self.n2p2_cell(),
                self.n2p2_atom(),
                self.n2p2_energy(),
                self.n2p2_charge(),
                'end \n' 
            ]
        else:
            block = [
                'begin',
                self.n2p2_cell(),
                self.n2p2_atom(),
                self.n2p2_energy(),
                self.n2p2_charge(),
                'end \n' 
            ]
        return list(flatten(block))