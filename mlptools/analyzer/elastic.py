import os
from glob import glob
import shutil
from typing import List
from mlptools.utils.utils import remove_empty_from_array


class ElasticConstantsCalculator:
    def __init__(self, path2n2p2_result, path2template) -> None:
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
            weights_file_dict[atomic_number_str] = sorted(glob(f'{self.path2n2p2_result}/weights.{atomic_number_str}.*'))

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


if __name__ == '__main__':
    path2n2p2_result = '/home/y1u0d2/result/n2p2/Si/103/dimer_10x_cutoff_6'
    path2template = '/home/y1u0d2/result/lammps/scripts/Si/elastic/template'

    ecc = ElasticConstantsCalculator(path2n2p2_result, path2template)
    ecc.setup(
        path2target=path2n2p2_result,
        atomic_number_list=[14]
    )