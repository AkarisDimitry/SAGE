import argparse
import os
from sage_lib import DFTPartition, SurfaceStatesGenerator  # Importing necessary classes

def generate_xyz_from_outcar(path, verbose=False):
    """
    Generate an XYZ file from a VASP OUTCAR file.

    Parameters:
    - path (str): Path to the VASP OUTCAR file.
    - verbose (bool): If True, prints additional information.
    """
    absolute_path = os.path.abspath(path)  # Convert to absolute path
    DP = DFTPartition(absolute_path)
    DP.readVASPFolder(v=verbose)
    DP.export_configXYZ()
    
def generate_vacancy(path, verbose=False):
    """
    Generate configurations with vacancies.

    Parameters:
    - path (str): Path to the VASP files directory.
    - verbose (bool): If True, prints additional information.
    """
    absolute_path = os.path.abspath(path)  # Convert to absolute path
    DP = DFTPartition(absolute_path)
    DP.readVASPFolder(v=verbose)
    DP.generateDFTVariants('Vacancy', [1])
    DP.exportVaspPartition()

def generate_disassemble_surface(path, steps=5, final_distance=5.0, atoms_to_remove=None, verbose=False):
    """
    Generate configurations for disassembling the surface.

    Parameters:
    - path (str): Path to the VASP files directory.
    - steps (int): Number of steps in the disassembly process.
    - final_distance (float): Final distance between layers or atoms.
    - atoms_to_remove (int or None): Specific number of atoms to remove.
    - verbose (bool): If True, prints additional information.
    """
    absolute_path = os.path.abspath(path)  # Convert to absolute path
    DP = SurfaceStatesGenerator(absolute_path)
    DP.readVASPFolder(v=verbose)
    DP.generate_disassemble_surface(steps=steps, final_distance=final_distance, atoms_to_remove=atoms_to_remove)
    DP.exportVaspPartition()

def main():
    parser = argparse.ArgumentParser(description='Tool for theoretical calculations in quantum mechanics and molecular dynamics.')
    subparsers = parser.add_subparsers(dest='command', help='Available sub-commands')

    # Sub-command to generate vacancy directory
    parser_vacancy = subparsers.add_parser('vacancy', help='Generate vacancy.')
    parser_vacancy.add_argument('--path', type=str, required=True, help='Path to the VASP files directory')   
    parser_vacancy.add_argument('--verbose', action='store_true', help='Display additional information')
    
    # Sub-command to generate XYZ file from an OUTCAR directory
    parser_xyz = subparsers.add_parser('xyz', help='Generate an XYZ file from an OUTCAR directory.')
    parser_xyz.add_argument('--path', type=str, required=True, help='Path to the OUTCAR directory')
    parser_xyz.add_argument('--verbose', action='store_true', help='Display additional information')
    parser_xyz.add_argument('--subfolders', action='store_true', help='Read all subfolders')

    # Sub-command to generate configurations for disassembling the surface
    parser_disassemble = subparsers.add_parser('disassemble', help='Generate configurations for disassembling the surface.')
    parser_disassemble.add_argument('--path', type=str, required=True, help='Path to the VASP files directory')
    parser_disassemble.add_argument('--steps', type=int, default=5, help='Number of steps in disassembly (default: 5)')
    parser_disassemble.add_argument('--final_distance', type=float, default=5.0, help='Final distance between layers or atoms (default: 5.0)')
    parser_disassemble.add_argument('--atoms_to_remove', type=int, help='Specific number of atoms to remove')
    parser_disassemble.add_argument('--verbose', action='store_true', help='Display additional information')

    args = parser.parse_args()

    # Handle execution based on the specified sub-command
    if args.command == 'xyz':
        generate_xyz_from_outcar(args.path, verbose=args.verbose)
    elif args.command == 'vacancy':
        generate_vacancy(args.path, verbose=args.verbose)
    elif args.command == 'disassemble':
        generate_disassemble_surface(args.path, steps=args.steps, final_distance=args.final_distance, atoms_to_remove=args.atoms_to_remove, verbose=args.verbose)

if __name__ == '__main__':
    main()