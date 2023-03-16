import collections

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