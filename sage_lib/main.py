import argparse
import os
from sage_lib import DFTPartition, SurfaceStatesGenerator, VacuumStatesGenerator  # Importing necessary classes

def generate_xyz_from_outcar(path, subfolders=False, verbose=False):
    """
    Generate an XYZ file from a VASP OUTCAR file.

    Parameters:
    - path (str): Path to the VASP OUTCAR file.
    - verbose (bool): If True, prints additional information.
    """
    DP = DFTPartition(path)
    read_files(partition=DP, path=path, subfolders=subfolders)
    DP.export_configXYZ()
    
def generate_vacancy(path, subfolders=False, verbose=False):
    """
    Generate configurations with vacancies.

    Parameters:
    - path (str): Path to the VASP files directory.
    - verbose (bool): If True, prints additional information.
    """
    DP = DFTPartition(path)
    read_files(partition=DP, path=path, subfolders=subfolders)
    
    DP.generateDFTVariants('Vacancy', [1])
    DP.exportVaspPartition()

def generate_disassemble_surface(path, steps=5, final_distance=5.0, atoms_to_remove=None, subfolders=False, verbose=False):
    """
    Generate configurations for disassembling the surface.

    Parameters:
    - path (str): Path to the VASP files directory.
    - steps (int): Number of steps in the disassembly process.
    - final_distance (float): Final distance between layers or atoms.
    - atoms_to_remove (int or None): Specific number of atoms to remove.
    - verbose (bool): If True, prints additional information.
    """
    SSG = SurfaceStatesGenerator(path)
    read_files(partition=SSG, path=path, subfolders=subfolders)

    SSG.generate_disassemble_surface(steps=steps, final_distance=final_distance, atoms_to_remove=atoms_to_remove)
    SSG.exportVaspPartition()

def generate_dimers(path=None, labels:list=None, steps:int=10, vacuum:float=18.0, subfolders=False, verbose=False):
    """
    Generate configurations for dimer search.

    Parameters:
    - path (str): Path to the VASP files directory (optional if labels are provided).
    - labels (list of str): List of atom labels (optional if path is provided).
    - steps (int): Number of steps in the dimer search.
    - vacuum (int): Specific vacuum distance.
    - verbose (bool): If True, prints additional information.
    """
    VSG = VacuumStatesGenerator(path)
    read_files(partition=VSG, path=path, subfolders=subfolders)

    if labels is not None: 
        VSG.generate_dimers(AtomLabels=labels, steps=steps )
    else: 
        VSG.generate_dimers(steps=steps )

    VSG.exportVaspPartition()

def generate_config(path:str=None, source:str=None, subfolders:bool=None, config_path:str=None, output_path:str=None, output_source:str=None, verbose=False):
    """
    Generate VASP partition and execution script for each container.

    Parameters:
    - path (str): Path to the VASP files directory.
    - config_path (str): Path to the configuration file.
    - output_path (str): Path for exporting VASP partition and scripts.
    - verbose (bool): If True, prints additional information.
    """
    DP = DFTPartition(path)
    read_files(DP, path=path, source=source, subfolders=subfolders, verbose=verbose)

    DP.readConfigSetup(config_path)
    if output_source.upper() == 'VASP':
        DP.exportVaspPartition(output_path, label='enumerate')
    
    DP.generate_execution_script_for_each_container([f'{n:03d}' for n, c in enumerate(DP.containers)], output_path)

    if verbose:
        print("Script and VASP partition generated successfully.")

def read_files(partition, path=None, source='VASP', subfolders=False, verbose=False):
    path = path if path is not None else partition.file_location
    absolute_path = os.path.abspath(path)
    partition._file_location = absolute_path

    if source.upper() == 'VASP':
        if subfolders:
            partition.readVASPSubFolder(absolute_path)
        else:
            partition.readVASPFolder(absolute_path)

    elif source.upper() == 'XYZ':
        if subfolders:
            partition.read_configXYZ(absolute_path) # IS NOT SUBFOLDER ! 
        else:
            partition.read_configXYZ(absolute_path)


def add_arguments(parser):
    parser.add_argument('--path', type=str, required=True, help='Path to the files directory')
    parser.add_argument('--verbose', action='store_true', help='Display additional information')
    parser.add_argument('--subfolders', default=False, action='store_true', help='Read from all subfolders under the specified path')
    parser.add_argument('--source', type=str, choices=['VASP', 'GAP', 'xyz'], default='VASP', help='Source of calculation from which the files originate: VASP, molecular_dynamics, or force_field (default: VASP)')

def main():
    parser = argparse.ArgumentParser(description='Tool for theoretical calculations in quantum mechanics and molecular dynamics.')
    subparsers = parser.add_subparsers(dest='command', help='Available sub-commands')

    # =========== Sub-command to generate vacancy directory ===========
    parser_vacancy = subparsers.add_parser('vacancy', help='Generate vacancy.')
    add_arguments(parser_vacancy)

    # =========== Sub-command to generate XYZ file from an OUTCAR directory ===========
    parser_xyz = subparsers.add_parser('xyz', help='Generate an XYZ file from an OUTCAR directory.')
    add_arguments(parser_xyz)

    # =========== Sub-command to generate configurations for disassembling the surface ===========
    parser_disassemble = subparsers.add_parser('disassemble', help='Generate configurations for disassembling the surface.')
    add_arguments(parser_disassemble)
    parser_disassemble.add_argument('--steps', type=int, default=5, help='Number of steps in disassembly (default: 5)')
    parser_disassemble.add_argument('--final_distance', type=float, default=5.0, help='Final distance between layers or atoms (default: 5.0)')
    parser_disassemble.add_argument('--atoms_to_remove', type=int, help='Specific number of atoms to remove')

    # =========== Sub-command: dimer ===========
    parser_dimer = subparsers.add_parser('dimer', help='Generate configurations for dimer search.')
    add_arguments(parser_dimer)
    parser_dimer.add_argument('--labels', nargs='+', help='List of atom labels for dimer search')
    parser_dimer.add_argument('--steps', type=int, default=10, help='Number of steps in the dimer search (default: 10)')
    parser_dimer.add_argument('--vacuum', type=int, default=18, help='Specific vacuum distance (default: 18)')

    # =========== Sub-comando para generar script ===========
    parser_script = subparsers.add_parser('config', help='Read Position data from "path", read Configurtion data from "config_path" and export to "output_path".')
    add_arguments(parser_script)
    parser_script.add_argument('--config_path', type=str, required=True, help='Path to the configuration file')
    parser_script.add_argument('--output_path', type=str, default='.', help='Path for exporting VASP partition and scripts')
    parser_script.add_argument('--output_source', type=str, default='VASP', help='Source for exporting partition and scripts')

    args = parser.parse_args()

    # Handle execution based on the specified sub-command
    if   args.command == 'xyz':
        generate_xyz_from_outcar(args.path, subfolders=args.subfolders, verbose=args.verbose)
    elif args.command == 'vacancy':
        generate_vacancy(args.path, subfolders=args.subfolders, verbose=args.verbose)
    elif args.command == 'disassemble':
        generate_disassemble_surface(args.path, steps=args.steps, final_distance=args.final_distance, atoms_to_remove=args.atoms_to_remove, subfolders=args.subfolders, verbose=args.verbose)
    elif args.command == 'dimer':
        generate_dimers(path=args.path, labels=args.labels, steps=args.steps, vacuum=args.vacuum, subfolders=args.subfolders, verbose=args.verbose)
    elif args.command == 'config':
        generate_config(path=args.path, source=args.source, subfolders=args.subfolders, config_path=args.config_path, 
                        output_path=args.output_path, output_source=args.output_source, verbose=args.verbose)

if __name__ == '__main__':
    main()