import numpy as np
import os
from glob import glob
import re
from typing import List

from ovito.io import import_file
from ovito.io.ase import ovito_to_ase

from mlptools.atoms.atom import MLPAtoms
from mlptools.io.parser import PWscfParser


def read_from_format(path2target:str=None, format:str=None) -> MLPAtoms:
    """Get MLPAtoms from some outputs. Currently support only PWscf

    Args:
        path2target (str, optional): path to output dir. Defaults to None.
        format (str, optional): Currently support only PWscf. Defaults to None.

    Raises:
        Exception: If unsupported format are selected.

    Returns:
        MLPAtoms: _description_
    """
    if format == 'espresso-in':
        parser = PWscfParser(path2target)
    else:
        raise Exception(f'{format} is not supported')
    
    return MLPAtoms(
            cell=parser.get_cell(),
            coord=parser.get_coord(),
            force=parser.get_force(),
            energy=parser.get_energy(),
            n_atoms=parser.get_n_atoms(),
            structure_id=parser.get_structure_id(),
            symbols=f'Si{parser.get_n_atoms()}'
        )

def read_from_lmp_dump(path2dump:str) -> List[MLPAtoms]:
    """get list of mlpatoms from lammps dumpfile

    Args:
        path2dump (str): path to dumpfile

    Raises:
        Exception: if frame is 0

    Returns:
        List[MLPAtoms]: _
    """
    pipeline = import_file(path2dump)
    print(f'Number of frames: {pipeline.source.num_frames}')
    if pipeline.source.num_frames == 0:
        raise Exception("There's no frame. pls see if the path ia correct")

    all_atoms = []
    # Loop over all frames of the sequence.
    for frame_index in range(pipeline.source.num_frames):
        data = pipeline.source.compute(frame_index)
        ase_atoms = ovito_to_ase(data)

        n_atoms = ase_atoms.get_positions().shape[0]
        mlpatoms = MLPAtoms(
            cell=ase_atoms.cell[:],
            coord=ase_atoms.get_positions(),
            force=None,
            energy=None,
            n_atoms=n_atoms,
            structure_id=None,
            symbols=f'Si{n_atoms}',
            frame=frame_index
        )
        all_atoms.append(mlpatoms)
    return all_atoms


def read_from_dp_data(path2target:str) -> List[MLPAtoms]:
    """Get list of MLPAtoms from deepmd data.

    Args:
        path2target (str): path to output dir. Defaults to None.

    Raises:
        Exception: If path is uncorrect

    Returns:
        List[MLPAtoms]: _description_
    """
    if len(glob(f'{path2target}/*.npy')) == 0:
        raise Exception("Invalid path")
    m = re.search('/mp-.*?/', path2target)
    id = '_'.join(m.group(0)[1:-1].split('_')[:-1])
    
    all_atoms = []
    all_box = np.load(os.path.join(path2target, 'box.npy'))
    all_force = np.load(os.path.join(path2target, 'force.npy'))
    all_energy = np.load(os.path.join(path2target, 'energy.npy'))
    all_coord = np.load(os.path.join(path2target, 'coord.npy'))
    for (box, coord, force, energy) in zip(all_box, all_coord, all_force, all_energy):
        n_atoms = int(len(coord)/3)
        atoms = MLPAtoms(
            cell=box.reshape(3,3),
            coord=coord.reshape(n_atoms, 3),
            force=force.reshape(n_atoms, 3),
            energy=energy[0] if type(energy) == np.ndarray else energy,
            structure_id=id,
            n_atoms=n_atoms,
            symbols=f'Si{n_atoms}'
        )
        all_atoms.append(atoms)
    return all_atoms