import argparse, os
from sage_lib import DFTPartition  # Make sure to import relevant classes

def generate_xyz_from_outcar(path, verbose=False):
    absolute_path = os.path.abspath(path)  # Convert the path to an absolute path
    DP = DFTPartition(absolute_path)
    DP.readVASPFolder(v=verbose)
    print(absolute_path)
    print(DP.containers[0].OutFileManager.AtomPositionManager[0])
    DP.export_configXYZ()

def main():
    parser = argparse.ArgumentParser(description='Tool for theoretical calculations in quantum mechanics and molecular dynamics.')
    subparsers = parser.add_subparsers(dest='command', help='Available sub-commands')

    # Sub-command to generate XYZ file from an OUTCAR directory
    parser_xyz = subparsers.add_parser('xyz', help='Generate an XYZ file from an OUTCAR directory.')
    parser_xyz.add_argument('--path', type=str, required=True, help='Path to the OUTCAR directory')
    parser_xyz.add_argument('--verbose', action='store_true', help='Display additional information')

    args = parser.parse_args()

    if args.command == 'xyz':
        generate_xyz_from_outcar(args.path, verbose=args.verbose)

if __name__ == '__main__':
    main()