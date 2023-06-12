from typing import List
from mlptools.atoms.atom import MLPAtoms
from glob import glob
import os
import pickle

ZERO_POINT_ENERGY = -630.972


def get_all_si_atoms(device='local') -> List[MLPAtoms]:
    if device == 'local':
        path2data = '/Users/y1u0d2/desktop/Lab/data/qe_data/Si'
    elif device == 'gpu':
        path2data = '/home/y1u0d2/data/qe_data/Si'
    else:
        raise Exception('Not supported device')

    all_data = []
    all_dirs = glob(f'{path2data}/*') + [f'{path2data}/amorphous']
    for mp_dir in all_dirs:
        # get all directories
        all_data += glob(f'{mp_dir}/*')

    all_atoms = []
    for path in all_data:
        path2pkl = os.path.join(path, 'mlpatoms.pkl')
        if not os.path.exists(path2pkl):
            continue
        with open(os.path.join(path, 'mlpatoms.pkl'), "rb") as f:
            all_atoms.append(pickle.load(f))
    
    return all_atoms