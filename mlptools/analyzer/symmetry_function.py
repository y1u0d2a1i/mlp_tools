from typing import Dict, List
import os
import pandas as pd
import numpy as np
import math

from mlptools.config.symmetry_function import RadialSymmetryFunctionConfig, AngularSymmetryFunctionConfig
from mlptools.utils.utils import remove_empty_from_array, log_decorator

class SymmetryFunctionValueReader():
    def __init__(self, path2target, path2nnpscaling=None):
        # check file existence
        if not os.path.exists(os.path.join(path2target, "atomic-env.G")):
            raise ValueError("atomic-env.G does not exist")
        if not os.path.exists(os.path.join(path2target, "input.nn")):
            raise ValueError("input.nn does not exist")
        
        path2nnpscaling = os.path.join(path2target, "nnp-scaling.log.0000") if path2nnpscaling is None else path2nnpscaling
        if not os.path.exists(path2nnpscaling):
            raise ValueError(f"nnp-scaling.log.0000 does not exist in {path2nnpscaling}")
        print("All files exist")
        
        self.path2target = path2target
        self.path2nnpscaling = path2nnpscaling

    def symmetry_function_values_dict(
            self, 
            atom_num_symbol_map: Dict[int, str]
        ) -> Dict[str, List[float]]:
        """対称性関数の値をatomic-env.Gから取得する

        Parameters
        ----------
        path2target : str
            atomic-env.Gまでへのパス
        atom_num_symbol_map : Dict[int, str]
            原子番号と元素記号の対応表
            (ex)
            atom_num_symbol_map = {
                14: "Si",
                8: "O"
            }

        Returns
        -------
        Dict[str, List[float]]
            元素記号をkeyとした対称性関数の値のリスト

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """

        # 初期化
        sf_val_list_dict = {}
        for _, atom_symbol in atom_num_symbol_map.items():
            sf_val_list_dict[atom_symbol] = []
        
        # 対称性関数の読み込み
        with open(os.path.join(self.path2target, "atomic-env.G"), "r") as f:
            lines = [s.strip() for s in f.readlines()]
        n_atoms = len(lines)
        for i, l in enumerate(lines):
            # show progress every 10000 lines
            if i % 10000 == 0:
                print(f"[PROGRESS] {i}/{n_atoms}")
            l_splitted = l.split()
            atomic_symbol = l_splitted[0]
            sf_vec = np.array(l_splitted[1:], dtype=float)

            if atomic_symbol in sf_val_list_dict.keys():
                sf_val_list_dict[atomic_symbol].append(sf_vec)
            else:
                raise ValueError("Unknown atomic number please check atom_num_symbol_map")
        return sf_val_list_dict


    def get_symmetry_function_columns(
            self, 
            number_of_sf_per_atom: int, 
            target_atom_symbol: str
        ) -> List[str]:
        """対称性関数のカラム名をlogとinput.nnから取得する
        logに書かれているlnからinput.nnの対応する行を取得する

        Parameters
        ----------
        path2target : str
            input.nnとnnp-scaling.log.0000までのパス
        number_of_sf_per_atom : int
            1原子あたりの対称性関数の数
        target_atom_symbol : str
            対象の元素記号

        Returns
        -------
        List[str]
            対称性関数のカラム名

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        """


        if not os.path.exists(os.path.join(self.path2target, 'input.nn')):
            raise ValueError("input.nn does not exist")
        
        with open(os.path.join(self.path2nnpscaling), mode='r') as f:
            scaling_log_lines = [s.strip() for s in f.readlines()]

        keyword = f"Short range atomic symmetry functions element"
        keyword_idx = None
        for line in scaling_log_lines:
            if keyword not in line:
                continue
                
            atom_symbol_in_line = remove_empty_from_array(line.split(' '))[-2]
            if target_atom_symbol == atom_symbol_in_line:
                keyword_idx = scaling_log_lines.index(line)
        if keyword_idx is None:
            raise ValueError(f"{target_atom_symbol} does not exist")
        
        sf_info_list_raw = scaling_log_lines[keyword_idx+4: keyword_idx+4+number_of_sf_per_atom]
        line_number_in_setting_file_list = [int(sf_info.split(' ')[-1]) for sf_info in sf_info_list_raw]
        
        with open(os.path.join(self.path2target, 'input.nn'), mode='r') as f:
            lines = [s.strip() for s in f.readlines()]

            target_sf_lines = [lines[idx-1] for idx in line_number_in_setting_file_list]

        sf_info_list_raw = scaling_log_lines[keyword_idx+4: keyword_idx+4+number_of_sf_per_atom]
        line_number_in_setting_file_list = [int(sf_info.split(' ')[-1]) for sf_info in sf_info_list_raw]
        sf_columns = ["_".join(target_sf_line.split(' ')[1:]) for target_sf_line in target_sf_lines]
        print(f"{target_atom_symbol} symmetry function columns are created")
        return sf_columns


    def read(
        self,
        atom_num_symbol_map: Dict[int, str],
        number_of_sf_per_atom: int
    ) -> Dict[str, pd.DataFrame]:
        """対称性関数の値を元素記号ごとにDataframeで返却する

        Parameters
        ----------
        atom_num_symbol_map : Dict[int, str]
            原子番号と元素記号の対応表
        number_of_sf_per_atom : int
            1原子あたりの対称性関数の数

        Returns
        -------
        Dict[str, pd.DataFrame]
            元素記号をkeyとした対称性関数の値のDataframe
        """
        sf_val_list_dict = self.symmetry_function_values_dict(
            atom_num_symbol_map=atom_num_symbol_map
        )
        sf_val_df_dict = {}
        for atom_symbol, sf_val_list in sf_val_list_dict.items():
            sf_columns = self.get_symmetry_function_columns(
                number_of_sf_per_atom=number_of_sf_per_atom,
                target_atom_symbol=atom_symbol
            )
            sf_val_df_dict[atom_symbol] = pd.DataFrame(sf_val_list, columns=sf_columns)
            print(f"Symmetry function dataframe of {atom_symbol} is created")
        return sf_val_df_dict


class SymmetryFunction:
    def cutoff_func1(self, r_ij, r_c):
        if r_ij <= r_c:
            return 0.5 * (math.cos(((np.pi * r_ij) / r_c)) + 1)
        elif r_ij > r_c:
            return 0

    def cutoff_func2(self, r_ij, r_c):
        if r_ij <= r_c:
            return (np.tanh(1-(r_ij/r_c)))**3
        elif r_ij > r_c:
            return 0

    def radial_symmetry_function_2(self, eta, r_ij, r_shift, r_cutoff, cut_func_type=1):
        cutoff_func = ''
        if cut_func_type == 1:
           cutoff_func = self.cutoff_func1(r_ij, r_cutoff)
        return math.exp(-1 * eta * (r_ij - r_shift) ** 2) * cutoff_func
    
    def ang_symmetry_function_3(self, theta, lambdas, zeta, is_degree=True):
        if is_degree:
            theta = np.radians(theta)
        return 2**(1-zeta) * (1 + lambdas * np.cos(theta))**zeta


class SymmetryFunctionSettingReader():
    SF_TYPE_IDX = 2

    def __init__(self, path2target, input_filename="input.nn") -> None:
        path2input = os.path.join(path2target, input_filename)
        if not os.path.exists(path2input):
            raise ValueError(f"{input_filename} does not exist")
        
        self.path2input = path2input
    

    def read_sf_setting_lines(self):
        with open(self.path2input, mode="r") as f:
            lines = [s.strip() for s in f.readlines()]

        sf_lines = list(filter(lambda s: s.startswith("symfunction_short"), lines))
        return sf_lines


    def get_radial_sf_lines(self):
        sf_lines = self.read_sf_setting_lines()
        sf_lines_splitted = [line.strip().split() for line in sf_lines]
        radial_sf_lines = list(filter(lambda s: s[self.SF_TYPE_IDX] == "2", sf_lines_splitted))
        return radial_sf_lines


    def get_angular_sf_lines(self):
        sf_lines = self.read_sf_setting_lines()
        sf_lines_splitted = [line.strip().split() for line in sf_lines]
        angular_sf_lines = list(filter(lambda s: s[self.SF_TYPE_IDX] == "3", sf_lines_splitted))
        return angular_sf_lines


    def _get_radial_sf_config(self) -> Dict[str, List[RadialSymmetryFunctionConfig]]:
        """n2p2の入力ファイルからRadialSymmetryFunctionConfigのリストをバリュー、結合をキーとする辞書を返却する

        Returns
        -------
        Dict[str, List[RadialSymmetryFunctionConfig]]
            _description_
        """
        radial_sf_lines = self.get_radial_sf_lines()
        radial_sf_config_list = []
        radial_sf_config_dict = {}

        # RadialSymmetryFunctionConfigのリストを作成
        print(f"Number of Radial Symmetry Funciton: {len(radial_sf_lines)}")
        for radial_sf_line in radial_sf_lines:
            radial_sf_config = RadialSymmetryFunctionConfig(
                bond=f"{radial_sf_line[1]}-{radial_sf_line[3]}", 
                eta=float(radial_sf_line[4]), 
                rs=float(radial_sf_line[5]), 
                rcut=float(radial_sf_line[6])
            )
            radial_sf_config_list.append(radial_sf_config)
        # bondをキーとした辞書を作成
        for radial_sf_config in radial_sf_config_list:
            if radial_sf_config_dict.get(radial_sf_config.bond) is None:
                radial_sf_config_dict[radial_sf_config.bond] = []
            else:
                radial_sf_config_dict[radial_sf_config.bond].append(radial_sf_config)
        return radial_sf_config_dict


    def _get_angular_sf_config(self) -> Dict[str, List[AngularSymmetryFunctionConfig]]:
        angular_sf_lines = self.get_angular_sf_lines()
        # AngularSymmetryFunctionConfigのリストを作成
        print(f"Number of Angular Symmetry Funciton: {len(angular_sf_lines)}")
        angular_sf_config_list = []
        for angular_sf_line in angular_sf_lines:
            angular_sf_config_list.append(
                AngularSymmetryFunctionConfig(
                    bond=f"{angular_sf_line[1]}-{angular_sf_line[3]}-{angular_sf_line[4]}",
                    eta=float(angular_sf_line[5]),
                    lambdas=float(angular_sf_line[6]),
                    zeta=float(angular_sf_line[7]),
                    rcut=float(angular_sf_line[8])
                )
            )
        # bondをキーとした辞書を作成
        angular_sf_config_dict = {}
        for angular_sf_config in angular_sf_config_list:
            if angular_sf_config_dict.get(angular_sf_config.bond) is None:
                angular_sf_config_dict[angular_sf_config.bond] = []
            else:
                angular_sf_config_dict[angular_sf_config.bond].append(angular_sf_config)
        return angular_sf_config_dict

    @log_decorator
    def get_radial_sf_config(self):
        radial_sf_config_dict = self._get_radial_sf_config()
        # print number of symmetry function for each bond
        for bond, radial_sf_config_list in radial_sf_config_dict.items():
            print(f"{bond} has {len(radial_sf_config_list)} symmetry functions")
        return radial_sf_config_dict
    
    @log_decorator
    def get_angular_sf_config(self):
        angular_sf_config_dict = self._get_angular_sf_config()
        # print number of symmetry function for each bond
        for bond, angular_sf_config_list in angular_sf_config_dict.items():
            print(f"{bond} has {len(angular_sf_config_list)} symmetry functions")
        return angular_sf_config_dict

class SymmetryFunctionVisualizer():
    def plot_sf_radial(self, config: RadialSymmetryFunctionConfig, rmax: float, ax, fontsize=12) -> None:
        """
        plot radial sf from param({eta: , rs: , rcut: })
        """
        sf = SymmetryFunction()
        r_ij = np.linspace(0, rmax, 100)
        ax.set_title(f'Radial symmetry functions: G2', fontsize=fontsize)
        ax.set_xlabel(f'r(Å)', fontsize=fontsize)
        # ax.set_ylabel(f'radial symmetry function : G2', fontsize=fontsize)
        ax.set_ylabel('$e^{-\eta}(R_{ij}-R_s)^2f_c(R_{ij})$', fontsize=fontsize)
        sf_value = [sf.radial_symmetry_function_2(eta=config.eta, r_ij=k, r_shift=config.rs, r_cutoff=config.rcut) for k in r_ij]
        ax.plot(r_ij, sf_value, label=f'η: {config.eta}, Rs: {config.rs}')


    def plot_sf_ang(self, config: AngularSymmetryFunctionConfig, ax, fontsize=12) -> None:
        """
        plot radial sf from param({eta: , rs: , rcut: })
        """
        sf = SymmetryFunction()
        theta = np.linspace(0, 360)
        ax.set_title(f'Angular symmetry functions: G3', fontsize=fontsize)
        ax.set_xlabel(r'$\theta(°)$', fontsize=fontsize)
        # ax.set_ylabel(f'Angular symmetry function : G3', fontsize=fontsize)
        ax.set_ylabel(r'$2^{1-\zeta} (1+\lambda \cos\theta_{ijk})^\zeta$', fontsize=fontsize)
        ax.set_xlim(0, 360)
        y = sf.ang_symmetry_function_3(theta=theta, lambdas=config.lambdas, zeta=config.zeta)
        ax.plot(theta, y, label=f'λ: {config.lambdas}, ζ: {config.zeta}')