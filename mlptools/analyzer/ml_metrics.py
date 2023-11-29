import pandas as pd
import numpy as np
import os
import glob
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def get_each_epoch_files(path2target):
    f_tests = glob.glob(f'{path2target}/testforces.*.out')
    f_trains = glob.glob(f'{path2target}/trainforces.*.out')
    e_tests = glob.glob(f'{path2target}/testpoints.*.out')
    e_trains = glob.glob(f'{path2target}/trainpoints.*.out')
    f_tests.sort()
    f_trains.sort()
    e_tests.sort()
    e_trains.sort()
    return f_tests, f_trains, e_tests, e_trains


def get_idx_from_columns(columns, lines):
    """
    docstring
    """
    for i, l in enumerate(lines):
        flag = False
        judge_list = [column in l for column in columns]
        if all(judge_list):
            idx = i
            break
    return idx
    

def convert_forceout_to_df(path2file):
    """
    docstring
    """
    _, conv_energy, conv_length = get_convert_factor('/'.join(path2file.split('/')[:-1]))
    columns = ['index_s', 'index_a', 'Fref', 'Fnnp']
    # idx = get_idx_from_columns(columns, lines=l_strip)
    # rows = l_strip[idx+2:]
    # rows = [row.split() for row in rows]
    # df = pd.DataFrame(data=rows, columns=columns)
    arr = np.loadtxt(path2file, skiprows=13)
    df = pd.DataFrame(data=arr, columns=columns)
    df = df.astype({
        'index_s': int, 
        'index_a': int, 
        'Fref': float, 
        'Fnnp': float
        })

    df['Fref_original'] = df['Fref'] * ( conv_length / conv_energy )
    df['Fnnp_original'] = df['Fnnp'] * ( conv_length / conv_energy )
    df = df.astype({
        'Fref_original': float,
        'Fnnp_original': float
        })

    return df


def convert_energyout_to_df(path2file):
    """
    docstring
    """
    mean_energy, conv_energy, _ = get_convert_factor('/'.join(path2file.split('/')[:-1]))
    std_energy = 1 / conv_energy

    # with open(path2file, mode='r') as f:
    #     l_strip = [s.strip() for s in f.readlines()]
        
    # columns = ['index', 'Eref', 'Ennp']
    # idx = get_idx_from_columns(columns, lines=l_strip)
    # rows = l_strip[idx+2:]
    # rows = [row.split() for row in rows]
    # df = pd.DataFrame(data=rows, columns=columns)
    arr = np.loadtxt(path2file, skiprows=13)
    df = pd.DataFrame(data=arr, columns=['index', 'Eref', 'Ennp'])
    df = df.astype({'index': int, 'Eref': float, 'Ennp': float})

    df['Eref_original'] = df['Eref'] * std_energy + mean_energy
    df['Ennp_original'] = df['Ennp'] * std_energy + mean_energy
    df = df.astype({'Eref_original': float, 'Ennp_original': float})

    return df


def calc_score(ref, pred) -> dict:
    dic = {
        'R2': r2_score(ref, pred),
        'RMSE': np.sqrt(mean_squared_error(ref, pred)),
        'MAE': mean_absolute_error(ref, pred)
    }
    return dic

def calc_force_score(epoch, data_type, force_df: pd.DataFrame)-> dict:
    ref, pred = force_df['Fref'], force_df['Fnnp']
    dic = calc_score(ref, pred)
    dic['epoch'] = epoch
    dic['data_type'] = data_type
    dic['type'] = 'force'
    return dic

def calc_energy_score(epoch, data_type, energy_df: pd.DataFrame) -> dict:
    ref, pred = energy_df['Eref'], energy_df['Ennp']
    dic = calc_score(ref, pred)
    dic['epoch'] = epoch
    dic['data_type'] = data_type
    dic['type'] = 'energy'
    return dic


def get_epoch(path):
    return path.split('/')[-1].split('.')[1]


def validate_epoch_file(f_test, f_train, e_test, e_train):
    if get_epoch(f_test) == get_epoch(f_train) == get_epoch(e_test) == get_epoch(e_train):
        return True
    else:
        return False
    
    
def get_all_score_df(path2target):
    f_tests, f_trains, e_tests, e_trains = get_each_epoch_files(path2target)
    num_epoch = len(f_tests)
    all_score_dict = {}
    for i, (f_test, f_train, e_test, e_train) in  enumerate(zip(f_tests, f_trains, e_tests, e_trains)):
        if validate_epoch_file(f_test, f_train, e_test, e_train):
            epoch = int(get_epoch(f_test))
            print(f'epoch: {epoch} out of {num_epoch}')
            f_test_score = calc_force_score(epoch=epoch, data_type='test', force_df=convert_forceout_to_df(f_test))
            f_train_score = calc_force_score(epoch=epoch, data_type='train', force_df=convert_forceout_to_df(f_train))
            e_test_score = calc_energy_score(epoch=epoch, data_type='test', energy_df=convert_energyout_to_df(e_test))
            e_train_score = calc_energy_score(epoch=epoch, data_type='train', energy_df=convert_energyout_to_df(e_train))
            idx_start = 4*i
            all_score_dict[idx_start] = f_test_score
            all_score_dict[idx_start+1] = f_train_score
            all_score_dict[idx_start+2] = e_test_score
            all_score_dict[idx_start+3] = e_train_score
    score_df = pd.DataFrame.from_dict(all_score_dict, orient='index')
    return score_df


def get_each_epoch_result_df(path2target, epoch):
    """
    epochごとの結果(Energy, force)取得
    """
    f_tests, f_trains, e_tests, e_trains = get_each_epoch_files(path2target)
    for f_test, f_train, e_test, e_train in  zip(f_tests, f_trains, e_tests, e_trains):
        if validate_epoch_file(f_test, f_train, e_test, e_train):
            if epoch == int(get_epoch(f_test)):
                f_test_df = convert_forceout_to_df(f_test)
                f_train_df = convert_forceout_to_df(f_train)
                e_test_df = convert_energyout_to_df(e_test)
                e_train_df = convert_energyout_to_df(e_train)
                break
    return f_test_df, f_train_df, e_test_df, e_train_df


def get_convert_factor(path2target):
    with open(os.path.join(path2target, 'input.nn'), mode='r') as f:
        lines = [s.strip() for s in f.readlines()]

    for l in lines:
        if 'mean_energy' in l: 
            mean_energy = float(l.split(' ')[-1])
        if 'conv_energy' in l:
            conv_energy = float(l.split(' ')[-1])
        if 'conv_length' in l:
            conv_length = float(l.split(' ')[-1])
    
    return mean_energy, conv_energy, conv_length