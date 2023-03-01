import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from glob import glob
import os
import random
from typing import List

from mlptools.io.read import read_from_format
from mlptools.atoms.atom import MLPAtoms
from mlptools.utils.constants import ZERO_POINT_ENERGY

def get_artificial_dimer(distance_lower_limit:float, distance_upper_limit:float, n_sample:int) -> List[MLPAtoms]:
    """get artificial dimer by fitting qe values with cubic spline

    Args:
        distance_lower_limit (float): lower limit of interatomic distance (ang)
        distance_upper_limit (float): upper limit of interatomic distance (ang)
        n_sample (int): number of artificial data

    Returns:
        List[MLPAtoms]: list of artificial atoms
    """
    if distance_lower_limit >= distance_upper_limit:
        raise Exception('lower limit is bigger than upper limit')

    path2dimer = '/Users/y1u0d2/desktop/Lab/result/qe/Si/mp-149_dimer/coord/04/result'

    all_atoms = []
    for d in glob(f'{path2dimer}/scf*'):
        try:
            atoms = read_from_format(d, format='espresso-in')
            all_atoms.append(atoms)
        except:
            continue

    print(f'Number of dimer atoms: {len(all_atoms)}')

    dimer_info = np.array([[atoms.energy, atoms.get_atomic_distance()] for atoms in all_atoms])
    dimer_df = pd.DataFrame(data=dimer_info, columns=['energy', 'distance']).sort_values(by='distance').reset_index(drop=True)
    dimer_df['shifted_energy'] = dimer_df['energy'] - 2 * ZERO_POINT_ENERGY

    interpolate_qe = dimer_df.query('distance > 2.31')[['distance', 'shifted_energy']].values

    artificial_data = np.array([
        [5.0, -0.45],
        [5.25, -0.27],
        [5.5, -0.15],
        [5.8, -0.05],
        [6.0, 0]
    ])
    interpolate_qe = np.concatenate((interpolate_qe, artificial_data), axis=0)
    cs = CubicSpline(interpolate_qe[:, 0], interpolate_qe[:, 1])

    sampled_distances = [round(random.uniform(distance_upper_limit, distance_lower_limit), 3) for _ in range(n_sample)]
    sampled_distances = list(set(sampled_distances))
    sampled_energy = cs(sampled_distances)
    sampled_force = cs(sampled_distances, 1)

    all_artificial_atoms = []
    for distance, energy, force in zip(sampled_distances, sampled_energy, sampled_force):
        all_artificial_atoms.append(
            MLPAtoms(
                cell=np.array([[15, 0, 0], [0, 15, 0], [0, 0, 15]]),
                coord=np.array([[7.5, 7.5, 7.5], [7.5+distance, 7.5, 7.5]]),
                energy=(2 * ZERO_POINT_ENERGY)+energy,
                force=np.array([[ force, 0, 0], [-force, 0, 0]]),
                n_atoms=2,
                structure_id='mp-149_dimer',
                symbols=f'Si2'
            )
        )
    
    return all_artificial_atoms