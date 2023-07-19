import os
from typing import List
import numpy as np
import pandas as pd


class SputteringYieldCalculator():
    def __init__(self, path2target:str, etch_file_name="etch.dat") -> None:
        self.path2target = path2target
        self.etch_file_name = etch_file_name

        if not os.path.exists(os.path.join(path2target, etch_file_name)):
            raise FileNotFoundError("The etch file is not found in the target directory.")
    

    def build_timestep_block(self) -> List[List[str]]:
        with open(os.path.join(self.path2target, self.etch_file_name), mode="r") as f:
            lines = [s.strip() for s in f.readlines()]
        
        idx_list = []
        for i, l in enumerate(lines):
            if 'ITEM: TIMESTEP' in l:
                idx_list.append(i)

        idx_list_shift = idx_list[1:]
        idx_list_shift.append(len(lines))

        block = []
        for idx1, idx2 in zip(idx_list, idx_list_shift):
            block.append(lines[idx1:idx2])
        return block


    def get_n_sputtered_atoms_with_timestep(self, target_atom_type:List[int]) -> pd.DataFrame:
        block = self.build_timestep_block()
        timestep_idx = 1
        num_sputter_atom_idx = 3
        sputtered_atom_info_idx = 9
        result = []
        for b in block:
            timestep = int(b[timestep_idx])
            num_sputter_atom = int(b[num_sputter_atom_idx])
            if num_sputter_atom > 0:
                print(f"timestep: {timestep}, num_sputter_atom: {num_sputter_atom}")
                sputtered_atom_list = b[sputtered_atom_info_idx:]
                sputtered_target_atom_list = list(filter(lambda x: int(x.split()[1]) in target_atom_type, sputtered_atom_list))
                num_sputter_atom = len(sputtered_target_atom_list)
                print(f"num_sputtered_target_atom: {num_sputter_atom}")
            result.append(np.array([timestep, num_sputter_atom]))
        result = np.array(result)
        return pd.DataFrame(result, columns=["timestep", "num_sputtered_atom"])