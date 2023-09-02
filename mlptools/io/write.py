from mlptools.atoms.atom import MLPAtoms
from mlptools.utils.utils import flatten
from typing import List
from abc import ABC, abstractmethod
from ase import Atoms
import os


def write_from_atoms(atoms: MLPAtoms, format: str) -> List[str]:
    if format == 'n2p2':
        writer = N2p2Writer(atoms)
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
    def __init__(self, atoms: Atoms, path2template: str) -> None:
        super().__init__(atoms)
        self.template = path2template
    

    def read_template(self):
        # read scf.in.template
        with open(os.path.join(self.template, 'scf.in'), 'r') as f:
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
        num_atoms = self.atoms.get_global_number_of_atoms()
        # change num of atoms
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
    def __init__(self, atoms, is_comment=True) -> None:
        self.is_comment = is_comment
        self.atoms = atoms
        

    def n2p2_comment(self):
        return f'comment {self.atoms.structure_id} .'

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
        force = self.atoms.force
        species = 'Si'
        for c, f in zip(coord, force):
            c = [str(i) for i in list(c)]
            f = [str(i) for i in list(f)]
            tmp_c = ' '.join(c)
            tmp_f = ' '.join(f)
            line.append(f'atom {tmp_c} {species} 0 0 {tmp_f}')
        return line
    
    def n2p2_energy(self):
        energy = self.atoms.energy
        return f'energy {energy}'
    
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