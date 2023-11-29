import os
from glob import glob
import shutil
from typing import List
from mlptools.utils.utils import remove_empty_from_array
import pandas as pd
from glob import glob
import os
from pydantic import BaseModel
from typing import Optional, Dict, Tuple, List

class MechanicalProps(BaseModel):
    c11: float
    c12: float
    c44: float
    bulk_modulus: float
    shear_modulus: float
    shear_modulus2: Optional[float]
    poisson_ratio: float

class ElasticConstantsCalculator:
    def __init__(self, path2n2p2_result=None, path2template=None) -> None:
        self.path2n2p2_result = path2n2p2_result
        self.path2template = path2template

    def change_nnpdir(self, lines, path2nnpdir):
        for i, l in enumerate(lines):
            if 'variable nnpDir' in l:
                dir_l = remove_empty_from_array(l.split(' '))
                dir_l[-1] = f'"{path2nnpdir}"'
                lines[i] = ' '.join(dir_l)
                break
        return lines

    def setup(self, path2target: str ,atomic_number_list: List[int]):
        path2elastic = os.path.join(path2target, 'elastic')
        if not os.path.exists(path2elastic):
            print(f"Create directory {path2elastic}")
            os.mkdir(path2elastic)

        atomic_number_list_preprocessed = [str(n).zfill(3) for n in atomic_number_list]
        weights_file_dict = {}
        for atomic_number_str in atomic_number_list_preprocessed:
            weights_file_dict[atomic_number_str] = sorted(glob(f'{self.path2n2p2_result}/weights.{atomic_number_str}.*.out'))

        for i in range(len(weights_file_dict[atomic_number_str])):
            weights_file_list = []
            for _, values in weights_file_dict.items():
                weights_file_list.append(values[i])
            
            epoch_set = {int(w.split('/')[-1].split('.')[-2]) for w in weights_file_list}
            if len(epoch_set) != 1:
                raise ValueError('epoch is not consistent')
            epoch = epoch_set.pop()
            print(f"epoch: {epoch}")
            path2epoch = os.path.join(path2elastic, f'e_{epoch}')

            # copy mechanical prop files
            if not os.path.exists(path2epoch): shutil.copytree(self.path2template, path2epoch)

            # copy n2p2 files
            shutil.copy(os.path.join(self.path2n2p2_result, 'input.nn'), path2epoch)
            shutil.copy(os.path.join(self.path2n2p2_result, 'scaling.data'), path2epoch)
            for weights_f in weights_file_list:
                shutil.copy(weights_f, path2epoch)
                print(f"[COPY] {weights_f} to {path2epoch}")
                atomic_str = os.path.basename(weights_f).split('.')[1]
                print(f"[RENAME] {os.path.basename(weights_f)} to weights.{atomic_str}.data")
                os.rename(os.path.join(path2epoch, os.path.basename(weights_f)), os.path.join(path2epoch, f"weights.{atomic_str}.data"))

            # change potential.mod
            with open(os.path.join(path2epoch, 'potential.mod'), mode='r') as f:
                lines = [s.strip() for s in f.readlines()]
            changed_lines = self.change_nnpdir(lines, path2epoch)
            with open(os.path.join(path2epoch, 'potential.mod'), mode='w') as f:
                f.write('\n'.join(changed_lines))
    
    
    def read_mechanical_props(self, lmp_log_lines) -> MechanicalProps:
        for l in lmp_log_lines:
            if 'Elastic Constant C11all' in l and 'GPa' in l:
                C11 = float(l.split(' ')[-2])
            
            if 'Elastic Constant C12all' in l and 'GPa' in l:
                C12 = float(l.split(' ')[-2])
            
            if 'Elastic Constant C44all' in l and 'GPa' in l:
                C44 = float(l.split(' ')[-2])

            if 'Bulk Modulus' in l and 'GPa' in l:
                bulk_modulus = float(l.split(' ')[-2])
            
            if 'Shear Modulus 1' in l and 'GPa' in l:
                shear_modulus = float(l.split(' ')[-2])
            
            if 'Shear Modulus 2' in l and 'GPa' in l:
                shear_modulus2 = float(l.split(' ')[-2])
            
            if 'Poisson Ratio' in l and '$' not in l:
                poisson_ratio = float(l.split(' ')[-1])
            
        return MechanicalProps(
            c11=C11,
            c12=C12,
            c44=C44,
            bulk_modulus=bulk_modulus,
            shear_modulus=shear_modulus,
            shear_modulus2=shear_modulus2,
            poisson_ratio=poisson_ratio
        )
    

    def get_mechanical_props(self, path: List[str]) -> Tuple[int, MechanicalProps]:
        with open(os.path.join(path, 'log.lammps'), mode='r') as f:
            lines = [s.strip() for s in f.readlines()]
        props = self.read_mechanical_props(lines)
        epoch = int(path.split('/')[-1].split('_')[-1])
        return epoch, props
    

    def get_mechanical_props_dict(self, elastic_dir_path_list: List[str]) -> Dict[int, MechanicalProps]:
        props_dict = {}
        for path in elastic_dir_path_list:
            epoch, props = self.get_mechanical_props(path)
            props_dict[epoch] = props
        return props_dict
    

    def get_mechanical_props_df(self, props_dict: Dict[int, MechanicalProps]) -> pd.DataFrame:
        mechanical_prop_df = pd.DataFrame.from_dict([{"epoch": k, **v.model_dump()} for k, v in props_dict.items()])
        # sort by epoch
        mechanical_prop_df.sort_values(by='epoch', inplace=True)
        mechanical_prop_df.reset_index(inplace=True, drop=True)
        return mechanical_prop_df



if __name__ == '__main__':
    path2n2p2_result = '/home/y1u0d2/result/n2p2/Si/103/dimer_10x_cutoff_6'
    path2template = '/home/y1u0d2/result/lammps/scripts/Si/elastic/template'

    ecc = ElasticConstantsCalculator(path2n2p2_result, path2template)
    ecc.setup(
        path2target=path2n2p2_result,
        atomic_number_list=[14]
    )