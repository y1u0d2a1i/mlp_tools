import collections
from mlptools.utils.constants import elements_dict



def log_decorator(func):
    """関数の前後にログを出力するデコレータ

    Parameters
    ----------
    func : _type_
        _description_
    """
    def wrapper(*args, **kwargs):
        print("*-" * 30)
        result = func(*args, **kwargs)
        print("*-" * 30)
        return result
    return wrapper


def get_param_idx(param, lines):
    """
    get param index from scf.in and scf.out
    """
    param_idx = None
    for i, l in enumerate(lines):
        if param in l:
            param_idx = i
    if param_idx is None:
        raise ValueError('invalid param')
    else:
        return param_idx

def remove_empty_from_array(arr: list) -> list:
    return list(filter(None, arr))

def flatten(l):
    """
    配列のを１次元にする
    """
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def get_mass_atom(species):
    mass_atom = elements_dict[species.upper()]
    if mass_atom:
        return mass_atom
    else:
        return 'none'

def get_atom_number(species):
    #  elm = Element(species)
    elm = elements_dict[species.upper()]
    atom_num = list(elements_dict.keys()).index(species.upper())
    try:
        return int(atom_num) + 1
    except:
        print('Invalid speacies')
    

def change_potential_lines(lmp_lines, cutoff, path2potential):
    for i, l in enumerate(lmp_lines):
        if "variable nnpCutoff" in l:
            lmp_lines[i] = f"variable nnpCutoff equal {cutoff}"
        if "variable nnpDir" in l:
            lmp_lines[i] = f'variable nnpDir string "{path2potential}"'
    return lmp_lines