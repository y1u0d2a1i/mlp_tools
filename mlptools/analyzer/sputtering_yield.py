import os
from typing import List, Optional
import numpy as np
import pandas as pd


class SputteringYieldCalculator():

    TIMESTEP_IDX = 1
    NUM_SPUTTER_ATOM_IDX = 3
    SPUTTERED_ATOM_INFO_IDX = 9

    def __init__(self, path2target:str, etch_file_name="etch.dat", inject_atom_every_timestep:Optional[int]=20000) -> None:
        self.path2target = path2target
        self.etch_file_name = etch_file_name
        self.inject_atom_every_timestep = inject_atom_every_timestep

        if not os.path.exists(os.path.join(path2target, etch_file_name)):
            raise FileNotFoundError("The etch file is not found in the target directory.")
        # 入射したイオン数
        self.n_injected_atoms = self.get_n_injected_atoms()
    

    def get_n_injected_atoms(self) -> int:
        block = self.build_timestep_block()
        last_timestep = int(block[-1][self.TIMESTEP_IDX])
        print(f"Number of total timestep: {last_timestep}")
        print(f"Inject atom every {self.inject_atom_every_timestep} timestep")
        print(f"Number of injected atoms: {last_timestep // self.inject_atom_every_timestep}")
        return last_timestep // self.inject_atom_every_timestep


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
        result = []
        for b in block:
            timestep = int(b[self.TIMESTEP_IDX])
            num_sputter_atom = int(b[self.NUM_SPUTTER_ATOM_IDX])
            if num_sputter_atom > 0:
                sputtered_atom_list = b[self.SPUTTERED_ATOM_INFO_IDX:]
                sputtered_target_atom_list = list(filter(lambda x: int(x.split()[1]) in target_atom_type, sputtered_atom_list))
                num_sputter_atom = len(sputtered_target_atom_list)
            result.append(np.array([timestep, num_sputter_atom]))
        result = np.array(result)
        return pd.DataFrame(result, columns=["timestep", "num_sputtered_atom"])
    

    def get_sputtering_yield_with_ion_dose(self, area:float=4.0725**2, num_injection:int=15, target_atom_type:List[int]=[1]) -> pd.DataFrame:
        sp_df = self.get_n_sputtered_atoms_with_timestep(target_atom_type=target_atom_type)
        print(f"Number of sputtered atoms: {sp_df['num_sputtered_atom'].sum()}")

        def get_num_inserted_atom(timestep, insert_atom_every_timestep):
            return (timestep // insert_atom_every_timestep) + 1

        def get_ion_dose(area, timestep, insert_atom_every_timestep):
            return get_num_inserted_atom(timestep, insert_atom_every_timestep) / area

        sp_df['num_inserted_atoms'] = sp_df['timestep'].apply(lambda x: get_num_inserted_atom(x, self.inject_atom_every_timestep))
        sp_df['ion_dose'] = sp_df['timestep'].apply(lambda x: get_ion_dose(area, x, self.inject_atom_every_timestep))

        max_num_inserted_atoms = sp_df['num_inserted_atoms'].max()

        sp_df_injection = sp_df.drop_duplicates(subset=['timestep'])
        sp_df_injection = sp_df_injection.query('timestep % @self.inject_atom_every_timestep == 0').copy()

        # get average number of sputtered atoms
        def get_averaged_num_sputtered_atoms(num_inserted_atoms, num_injection):
            lower = num_inserted_atoms - num_injection if num_inserted_atoms - num_injection > 0 else 0
            upper = num_inserted_atoms + num_injection if num_inserted_atoms + num_injection < max_num_inserted_atoms else max_num_inserted_atoms
            return sp_df.query('num_inserted_atoms >= @lower and num_inserted_atoms <= @upper')['num_sputtered_atom'].sum() / (upper - lower + 1)

        sp_df_injection['num_sputtered_atom_avg'] = sp_df_injection['num_inserted_atoms'].apply(lambda x: get_averaged_num_sputtered_atoms(x, num_injection))

        return sp_df_injection
    

    def get_injected_and_sputtered_atoms(self, target_atom_type:List[int]=[1]):
        sp_df = self.get_n_sputtered_atoms_with_timestep(target_atom_type=target_atom_type)
        print(f"Number of sputtered atoms: {sp_df['num_sputtered_atom'].sum()}")
        max_timestep = sp_df["timestep"].max()
        num_injection = int(np.ceil(max_timestep/self.inject_atom_every_timestep))
        num_injected_sputtered_atoms = {
            'num_injected_atoms': [],
            'num_sputtered_atoms': []
        }
        for i in range(num_injection):
            sum_up_timestep_inteval = (self.inject_atom_every_timestep * i, self.inject_atom_every_timestep * (i + 1))
            num_injected_atoms = i+1
            num_sputtered_atom_per_injection = sp_df.query(f"{sum_up_timestep_inteval[0]} <= timestep < {sum_up_timestep_inteval[1]}")['num_sputtered_atom'].sum()
            # print(f"Sum up timestep interval: {sum_up_timestep_inteval}")
            # print(f"Number of injected atoms: {num_injected_atoms}, sputtered atoms: {num_sputtered_atom_per_injection}")

            num_injected_sputtered_atoms['num_injected_atoms'].append(num_injected_atoms)
            num_injected_sputtered_atoms['num_sputtered_atoms'].append(num_sputtered_atom_per_injection)
        num_injected_sputtered_atoms_df = pd.DataFrame(num_injected_sputtered_atoms)
        return num_injected_sputtered_atoms_df