import numpy as np
import os
from glob import glob
import re
from typing import List

from ovito.io import import_file
from ovito.io.ase import ovito_to_ase
from ase import Atoms

from mlptools.atoms.atom import MLPAtoms
from mlptools.io.parser import PWscfParser
from mlptools.io.parser import ASEParser


def read_from_format(path2target:str=None, format:str=None, structure_id=None, ase_atoms: Atoms=None, is_validate_strict=True, has_calculator=True) -> MLPAtoms:
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
        parser = PWscfParser(
            path_to_target=path2target, 
            structure_id=structure_id,
            is_validate_strict=is_validate_strict
        )
    elif format == 'ase':
        parser = ASEParser(
            ase_atoms=ase_atoms,
            structure_id=structure_id,
            has_calculator=has_calculator
        )
    else:
        raise Exception(f'{format} is not supported')
    
    return MLPAtoms(
            cell=parser.get_cell(),
            coord=parser.get_coord(),
            force=parser.get_force(),
            energy=parser.get_energy(),
            n_atoms=parser.get_n_atoms(),
            total_magnetization=parser.get_total_magnetization(),
            structure_id=parser.get_structure_id(),
            symbols=parser.get_symbol(),
            ase_atoms=parser.get_ase_atoms(),
            path=path2target,
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


def read_from_dp_data(path2target:str, additional_info=None) -> List[MLPAtoms]:
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
            symbols=f'Si{n_atoms}',
            additional_info=additional_info
        )
        all_atoms.append(atoms)
    return all_atoms


def read_from_n2p2_data(path2target:str, data_filename:str="input.data") -> List[MLPAtoms]:
    def get_lattice_from_n2p2_data(path2target):
        with open(os.path.join(path2target, data_filename), mode="r") as f:
            lines = [s.strip() for s in f.readlines()]
        block = []
        block_start = []
        block_end = []
        for i, line in enumerate(lines):
            if line == "begin":
                block_start.append(i)
            elif line == "end":
                block_end.append(i)

        for i in range(len(block_start)):
            block.append(lines[block_start[i] : block_end[i] + 1])
        return block
    
    blocks = get_lattice_from_n2p2_data(path2target)

    all_mlpatoms = []
    for i, block in enumerate(blocks):
        # show progress
        if i % 1000 == 0:
            print(f"{i} / {len(blocks)}")
            
        comment = [l for l in block if l.startswith("comment")]
        structure_id = comment[0].split(' ')[1]

        lattice = [l for l in block if l.startswith("lattice")]
        lattice = np.array([list(filter(None, l.split(' ')))[1:] for l in lattice], dtype=float)
        
        atoms = [l for l in block if l.startswith("atom")]
        coord = []
        force = []
        chemical_symbols = []
        for atom in atoms:
            splitted_atom = list(filter(None, atom.split(' ')))
            coord.append(splitted_atom[1:4])
            force.append(splitted_atom[-3:])
            chemical_symbols.append(splitted_atom[4])
        coord = np.array(coord, dtype=float)
        force = np.array(force, dtype=float)
        
        energy = float([l for l in block if l.startswith("energy")][0].split(' ')[-1])
        charge = float([l for l in block if l.startswith("charge")][0].split(' ')[-1])

        mlpatom = MLPAtoms(
            cell=lattice,
            coord=coord,
            force=force,
            energy=energy,
            n_atoms=len(coord),
            structure_id=structure_id,
            symbols=chemical_symbols,
        )
        all_mlpatoms.append(mlpatom)
    return all_mlpatoms