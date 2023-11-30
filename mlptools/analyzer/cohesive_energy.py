import os
from mlptools.utils.utils import change_potential_lines, log_decorator


class CohesiveEnergyCalculator():
    def __init__(self, path2template, path2potential, path2target, cutoff=5.0) -> None:
        self.cutoff = cutoff
        self.path2template = path2template
        self.path2potential = path2potential
        self.path2target = path2target


    def load_template(self):
        # load template
        with open(os.path.join(self.path2template, "single", "in.single"), 'r') as f:
            single_lines = [s.strip() for s in f.readlines()]

        with open(os.path.join(self.path2template, "minimize", "in.minimize"), 'r') as f:
            minimize_lines = [s.strip() for s in f.readlines()]
        
        return single_lines, minimize_lines
    

    def get_lmp_lines(self, single_lines, minimize_lines, path2potential):
        changed_single_lines = change_potential_lines(single_lines, self.cutoff, path2potential)
        changed_minimize_lines = change_potential_lines(minimize_lines, self.cutoff, path2potential)
        return changed_single_lines, changed_minimize_lines
    
    @log_decorator
    def setup(self):
        single_lines, minimize_lines = self.load_template()
        changed_single_lines, changed_minimize_lines = self.get_lmp_lines(
            single_lines=single_lines,
            minimize_lines=minimize_lines,
            path2potential=self.path2potential
        )
        # write to file
        path2single = os.path.join(self.path2target, "single")
        path2minimize = os.path.join(self.path2target, "minimize")
        os.makedirs(path2single, exist_ok=True)
        os.makedirs(path2minimize, exist_ok=True)
        with open(os.path.join(path2single, "in.single"), 'w') as f:
            f.write("\n".join(changed_single_lines))
        print(f"write to {os.path.join(path2single, 'in.single')}")

        with open(os.path.join(path2minimize, "in.minimize"), 'w') as f:
            f.write("\n".join(changed_minimize_lines))
        print(f"write to {os.path.join(path2minimize, 'in.minimize')}")

    
class CohesiveEnergyReader():
    def __init__(self, num_atoms:int=8000) -> None:
        self.NUM_ATOMS = num_atoms

    def read(self, path2target):
        path2single = os.path.join(path2target, "single")
        path2minimize = os.path.join(path2target, "minimize")

        if not os.path.exists(os.path.join(path2single, "log.lammps")):
            raise FileNotFoundError(f"log.lammps is not found in {path2single}")
        if not os.path.exists(os.path.join(path2minimize, "log.lammps")):
            raise FileNotFoundError(f"log.lammps is not found in {path2minimize}")


        # read single energy
        with open(os.path.join(path2single, "log.lammps"), 'r') as f:
            log_single_lines = [s.strip() for s in f.readlines()]
        

        energy_idx = None
        for i, l in enumerate(log_single_lines):
            if "TotEng" in l:
                energy_idx = i+1

        single_energy = float(log_single_lines[energy_idx].split()[-2])

        # read minimized energy and number of atoms
        with open(os.path.join(path2minimize, "log.lammps"), 'r') as f:
            log_minimize_lines = [s.strip() for s in f.readlines()]

        for i, l in enumerate(log_minimize_lines):
            if "Energy initial, next-to-last, final" in l:
                minimized_energy = float(log_minimize_lines[i+1].split()[-1])

        print("*" * 50)
        cohesive_energy = (minimized_energy - self.NUM_ATOMS * single_energy) / self.NUM_ATOMS
        print(f"Cohesive energy: {cohesive_energy} eV/atom")
        return cohesive_energy