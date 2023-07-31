import numpy as np
import pandas as pd

from ase import Atoms


class DepthProfileCalculator():
    ANG3_TO_CM3 = 1.0e24
    def __init__(self):
        pass

    def get_depth_profile(
            self,
            atoms: Atoms, 
            atom_type: int, 
            x_width:float, 
            y_width: float, 
            z_width: float, 
            upper_limit: float,
            step: float=0.5,
            moving_average_with_depth_interval: int=3) -> pd.DataFrame:
        type_idx_in_atoms = np.where(atoms.symbols.numbers == atom_type)[0]
        position_df = pd.DataFrame(data=atoms.positions[type_idx_in_atoms], columns=['x', 'y', 'z'])
        z_position = position_df['z'].values

        # convert z-position to depth
        depth = z_position - z_width
        upper = upper_limit - z_width
        linspace = np.linspace(upper, -z_width, np.ceil(upper_limit / step).astype(int))
        # linspace = np.linspace(upper, -z_width, n_space + 1)

        width = step
        density_list = []
        for i in linspace:
            n_atom_in_range = ((i - width/2 < depth) & (depth < i + width/2)).sum()
            vol = x_width * y_width * width
            density = n_atom_in_range / vol * self.ANG3_TO_CM3 # cm^-3
            # density = n_atom_in_range / vol # ang^-3
            density_list.append(density)

        df = pd.DataFrame(data=np.stack([linspace, density_list]).T, columns=['linspace', 'density'])
        df['ma'] = df['density'].rolling(int(moving_average_with_depth_interval / step + 1), center=True).mean()
        df.dropna(axis=0, inplace=True)
        return df