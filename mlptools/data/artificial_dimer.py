import numpy as np
from mlptools.atoms.atom import MLPAtoms

def get_artificial_dimer():
    energy_zero_point = -1260.14108
    artificial_data = [
        {
            'box': np.array([[15, 0, 0, 0, 15, 0, 0, 0, 15]]),
            'force': np.array([[ 0.76309001, 0, 0, -0.76309001, 0, 0]]),
            'energy': energy_zero_point - 0.45,
            'coord': np.array([[7.5, 7.5, 7.5, 7.5+5.0, 7.5, 7.5]])
        },
        {
            'box': np.array([[15, 0, 0, 0, 15, 0, 0, 0, 15]]),
            'force': np.array([[ 0.61191261, 0, 0, -0.61191261, 0, 0]]),
            'energy': energy_zero_point - 0.27,
            'coord': np.array([[7.5, 7.5, 7.5, 7.5+5.25, 7.5, 7.5]])
        },
        {
            'box': np.array([[15, 0, 0, 0, 15, 0, 0, 0, 15]]),
            'force': np.array([[ 0.38925955, 0, 0, -0.38925955, 0, 0]]),
            'energy': energy_zero_point - 0.15,
            'coord': np.array([[7.5, 7.5, 7.5, 7.5+5.5, 7.5, 7.5]])
        },
        {
            'box': np.array([[15, 0, 0, 0, 15, 0, 0, 0, 15]]),
            'force': np.array([[ 0.28096285, 0, 0, -0.28096285, 0, 0]]),
            'energy': energy_zero_point - 0.05,
            'coord': np.array([[7.5, 7.5, 7.5, 7.5+5.8, 7.5, 7.5]])
        },
        {
            'box': np.array([[15, 0, 0, 0, 15, 0, 0, 0, 15]]),
            'force': np.array([[ 0.22061748, 0, 0, -0.22061748, 0, 0]]),
            'energy': energy_zero_point,
            'coord': np.array([[7.5, 7.5, 7.5, 7.5+6.0, 7.5, 7.5]])
        },
    ]
    mlpatoms = []
    for data in artificial_data:
        mlpatoms.append(
            MLPAtoms(
                cell=data['box'],
                coord=data['coord'],
                energy=data['energy'],
                force=data['force'],
                n_atoms=2,
                structure_id='mp-149_dimer',
                symbols=f'Si2'
                )
        )
    return mlpatoms