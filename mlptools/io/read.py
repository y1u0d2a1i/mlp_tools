from mlptools.atoms.atom import MLPAtoms
from mlptools.io.parser import PWscfParser


def read_from_format(path2target=None, format=None):
    if format == 'espresso-in':
        parser = PWscfParser(path2target)
    else:
        raise Exception(f'{format} is not supported')
    
    return MLPAtoms(
            cell=parser.get_cell(),
            coord=parser.get_coord(),
            force=parser.get_force(),
            energy=parser.get_energy(),
            n_atoms=parser.get_n_atoms(),
            structure_id=parser.get_structure_id(),
            symbols=f'Si{parser.get_n_atoms()}'
        )