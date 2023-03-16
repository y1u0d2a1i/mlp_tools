from mlptools.atoms.atom import MLPAtoms
from mlptools.utils.utils import flatten

class N2p2Writer:
    def __init__(self, atoms, is_comment=True) -> None:
        self.is_comment = is_comment
        self.atoms = atoms
        

    def n2p2_comment(self):
        return f'comment {self.atoms.path} .'

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
        coord = self.atoms.cell
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
    
    def n2p2_block(self):
        if self.is_comment:
            block = [
                'begin',
                self.n2p2_comment,
                self.n2p2_cell,
                self.n2p2_atom,
                self.n2p2_energy,
                self.n2p2_charge,
                'end \n' 
            ]
        else:
            block = [
                'begin',
                self.n2p2_cell,
                self.n2p2_atom,
                self.n2p2_energy,
                self.n2p2_charge,
                'end \n' 
            ]
        return flatten(block) 