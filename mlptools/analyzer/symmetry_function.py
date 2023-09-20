from typing import Dict, List
import os
import pandas as pd

from mlptools.utils.utils import remove_empty_from_array


class SymmetryFunctionValueReader():
    def __init__(self, path2target):
        # check file existence
        if not os.path.exists(os.path.join(path2target, "function.data")):
            raise ValueError("function.data does not exist")
        if not os.path.exists(os.path.join(path2target, "input.nn")):
            raise ValueError("input.nn does not exist")
        if not os.path.exists(os.path.join(path2target, "nnp-scaling.log.0000")):
            raise ValueError("nnp-scaling.log.0000 does not exist")
        print("All files exist")
        
        self.path2target = path2target

    def symmetry_function_values_dict(
            self, 
            atom_num_symbol_map: Dict[int, str]
        ) -> Dict[str, List[float]]:
        """対称性関数の値をfunction.dataから取得する

        Parameters
        ----------
        path2target : str
            function.dataまでへのパス
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
        if not os.path.exists(os.path.join(self.path2target, "function.data")):
            raise ValueError("function.data does not exist")
        
        with open(os.path.join(self.path2target, "function.data"), "r") as f:
            lines = [s.strip() for s in f.readlines()]
        

        idx_list = []
        for i, line in enumerate(lines):
            if len(remove_empty_from_array(line.split(' '))) == 1:
                idx_list.append(i)

        sf_val_list = []
        for i, idx in enumerate(idx_list):
            try:
                block = lines[idx+1:idx_list[i+1]-1]
            except:
                block = lines[idx+1:-1]
            
            for l in block:
                sf_val_list.append(list(map(float, remove_empty_from_array(l.split(' ')))))
        
        # 初期化
        sf_val_list_dict = {}
        for _, atom_symbol in atom_num_symbol_map.items():
            sf_val_list_dict[atom_symbol] = []
        
        # 1つ目の要素はintなのでintに変換し、要素ごとに分ける
        for sf_val in sf_val_list:
            if int(sf_val[0]) in atom_num_symbol_map.keys():
                sf_val_list_dict[atom_num_symbol_map[int(sf_val[0])]].append(sf_val[1:])
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
        if not os.path.exists(os.path.join(self.path2target, 'nnp-scaling.log.0000')):
            raise ValueError("nnp-scaling.log.0000 does not exist")

        if not os.path.exists(os.path.join(self.path2target, 'input.nn')):
            raise ValueError("input.nn does not exist")
        
        with open(os.path.join(self.path2target, 'nnp-scaling.log.0000'), mode='r') as f:
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