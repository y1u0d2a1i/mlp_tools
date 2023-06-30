from mlptools.io.read import read_from_format

path2scf = "/Users/y1u0d2/desktop/Lab/data/qe_data/SiO2/mp-546794/result/scf978"
mlpatoms = read_from_format(path2scf, format='espresso-in')
print(mlpatoms.get_ase_atoms())