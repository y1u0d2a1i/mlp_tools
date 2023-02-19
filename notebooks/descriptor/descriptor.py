import numpy as np
import pandas as pd
import os
from glob import glob
from deepmd.infer import DeepPot

from mlptools.io.read import read_from_dp_data, read_from_lmp_dump
from mlptools.descriptor.dp import get_descriptor_vector, get_predictions

path2dump = '/home/y1u0d2/result/lammps/scripts/Si/sputtering/deepmd/model26/0deg_100eV/dump.lammpstrj'
path2model = '/home/y1u0d2/result/deepmd/Si/27/graph.pb'

path2data = '/home/y1u0d2/result/deepmd/Si/27/data'
path2test = os.path.join(path2data, 'test')
path2train = os.path.join(path2data, 'train')

all_atoms = []
for path in glob(f'{path2test}/mp*'):
    all_atoms += read_from_dp_data(os.path.join(path, 'set.000'))

for path in glob(f'{path2train}/mp*'):
    all_atoms += read_from_dp_data(os.path.join(path, 'set.000'))


all_descriptor = []
structure_id_list = []
model = DeepPot(path2model)

for i, atoms in enumerate(all_atoms):
    print(i ,atoms.structure_id, atoms.n_atoms)
    descriptor = get_descriptor_vector(atoms, model=model)

    with open(os.path.join(path2data, 'structure_id.txt'), mode='a') as f:
        f.write("\n".join([atoms.structure_id] * atoms.n_atoms)+"\n")
    
    with open(os.path.join(path2data, 'descriptor.txt'), mode='a') as f:
        np.savetxt(f, descriptor[0])


# structure_id_list = np.hstack(np.array(structure_id_list))
# all_descriptor = np.vstack(np.array(all_descriptor))

# descriptor_df = pd.DataFrame(data=all_descriptor, columns=[f'dp_{i}' for i in range(all_descriptor.shape[1])])
# descriptor_df['structure_id'] = structure_id_list

# path2save = '/home/y1u0d2/result/deepmd/Si/27/data'
# descriptor_df.to_csv(os.path.join(path2save, 'descriptor.csv'), index=None)