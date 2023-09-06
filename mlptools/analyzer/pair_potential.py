from mlptools.io.read import read_from_format
from typing import List
import pandas as pd
import os
from ase.io.espresso import read_espresso_in


class PairPotentialAnalyzer():
    def __init__(self) -> None:
        self.RY2EV = 13.605703976


    def validate_espresso_out(self, lines: List[str]) -> bool:
        for line in lines:
            if "!    total energy" in line:
                return True
        return False
    

    def get_from_espresso(self, atoms_dirs: List[str]) -> pd.DataFrame:
        # """Quantum espressoの計算結果から、ペアポテンシャルを取得する

        # Parameters
        # ----------
        # atoms_dirs : List[str]
        #     Quantum espressoの計算結果が格納されたディレクトリのリスト
        # """
        pair_potential_dict = {
            "distance": [],
            "energy": []
        }
        for atom_d in atoms_dirs:
            try:
                atoms = read_from_format(atom_d, format="espresso-in")
                pair_potential_dict["distance"].append(atoms.get_atomic_distance())
                pair_potential_dict["energy"].append(atoms.energy)
            except Exception as e:
                print(e)
                continue

        pair_potential_df = pd.DataFrame.from_dict(pair_potential_dict)
        pair_potential_df.sort_values(by="distance", inplace=True)
        pair_potential_df.reset_index(inplace=True, drop=True)
        return pair_potential_df
    

    def get_from_espresso_mag(self, atoms_dirs: List[str]) -> pd.DataFrame:
        pair_potential_dict = {
            "distance": [],
            "energy": []
        }

        for scf_d in atoms_dirs:
            with open(os.path.join(scf_d, 'scf.out')) as f:
                scf_out_lines = [s.strip() for s in f.readlines()]
            if not self.validate_espresso_out(scf_out_lines):
                print(f"scf.out in {scf_d} is not valid")
                continue
            
            # get atoms from input
            atoms = read_espresso_in(os.path.join(scf_d, 'scf.in'))
            distance = atoms.get_distance(0, 1)

            # get energy
            for line in scf_out_lines:
                if "!    total energy" in line:
                    energy = float(line.split()[-2]) * self.RY2EV
                    break
            
            pair_potential_dict["distance"].append(distance)
            pair_potential_dict["energy"].append(energy)

        pair_potential_df = pd.DataFrame.from_dict(pair_potential_dict)
        pair_potential_df.sort_values(by="distance", inplace=True)
        pair_potential_df.reset_index(inplace=True, drop=True)
        return pair_potential_df