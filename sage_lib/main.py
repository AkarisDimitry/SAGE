import argparse
import os
from sage_lib.partition.Partition import Partition 
from sage_lib.input.structure_handling_tools.AtomPosition import AtomPosition 
from sage_lib.output.EigenvalueFileManager import EigenvalueFileManager 
from sage_lib.output.DOSManager import DOSManager 

from sage_lib.output.OutFileManager import OutFileManager 

def generate_vacancy(path:str, source:str='VASP', subfolders:bool=False, verbose:bool=False):
    """
    Generate configurations with vacancies.

    Parameters:
    - path (str): Path to the VASP files directory.
    - verbose (bool): If True, prints additional information.
    """
    PT = Partition(path)
    PT.read_files( file_location=path, source=source, subfolders=subfolders, verbose=verbose)
    PT.generateDFTVariants('Vacancy', [1])
    PT.export_files(file_location=path, source=source, label=None, verbose=verbose)

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

def generate_config(path: str = None, source: str = None, subfolders: bool = None, 
                    config_path: str = None, config_source: str = None, 
                    output_path: str = None, output_source: str = None, verbose: bool = False):
    """
    Generates a configuration by reading, processing, and exporting files.

    This function orchestrates the workflow of partitioning data, reading configuration setup, 
    and exporting files with enumeration. It provides verbose output for debugging and tracking.

    Parameters:
    - path (str): Path to the data files.
    - source (str): Source identifier for the data files.
    - subfolders (bool): Flag to include subfolders in the data reading process.
    - config_path (str): Path to the configuration setup files.
    - config_source (str): Source identifier for the configuration files.
    - output_path (str): Path for exporting the processed files.
    - output_source (str): Source identifier for the exported files.
    - verbose (bool): Flag for verbose output.

    Returns:
    None
    """

    PT = Partition(path)
    PT.read_files(file_location=path, source=source, subfolders=subfolders, verbose=verbose)
    PT.read_Config_Setup(file_location=config_path, source=config_source, verbose=verbose)
    PT.export_files(file_location=output_path, source=output_source, label='enumerate', verbose=verbose)

    if verbose:
        print(f">> Config generated successfully.")
        print(f"Position: {path}({source})(subfolders: {subfolders}) + \n InputFiles: {config_path}({output_path}) >> Output: \n {output_path}({output_source})")

def generate_band_calculation(path:str, points:int, special_points:str, source:str=None, subfolders:bool=False, output_path:str=None, verbose:bool=False):
    """
    Generate and export band structure calculation files.

    This function creates the necessary files for performing band structure calculations using Density Functional Theory (DFT) data. It sets up the calculation parameters and exports them in a format suitable for VASP.

    Parameters:
    path (str): Path to the directory containing VASP files.
    points (int): Number of k-points in each segment of the band path.
    special_points (str): String representing high-symmetry points in the Brillouin zone.
    source (str, optional): Source of the files (default is None, typically set to 'VASP').
    subfolders (bool, optional): Whether to include subfolders in the search (default False).
    output_path (str, optional): Directory path where the output files will be saved.
    verbose (bool, optional): If True, provides detailed output during execution.

    Returns:
    None
    """
    DP = Partition(path)
    read_files(partition=DP, path=path, source=source, subfolders=subfolders, verbose=verbose)

    DP.generate_variants('band_structure', values=[{'points':points, 'special_points':special_points}])
    DP.exportVaspPartition()

def generate_json_from_bands(path:str, fermi:float, source:str=None, subfolders:bool=False, output_path:str=None, verbose:bool=False):
    """
    Generate a JSON file from band structure data.

    This function reads the band structure data from VASP output files, processes it, and exports it to a JSON file. This is useful for further analysis or visualization of the band structure.

    Parameters:
    path (str): Path to the directory containing VASP files.
    fermi (float): Fermi level energy. If not provided, it will be read from the DOSCAR file.
    source (str, optional): Source of the files ('VASP' is a common option).
    subfolders (bool, optional): Whether to include subfolders in the search (default False).
    output_path (str, optional): Directory path where the JSON file will be saved.
    verbose (bool, optional): If True, provides detailed output during execution.

    Returns:
    None
    """
    if source.upper() == 'VASP':
        # read fermi level from DOSCAR
        if fermi is None:
            # === read DOCAR === #
            DM = DOSManager(path + "/DOSCAR")
            DM.read_DOSCAR()
            fermi = DM.fermi

        # === read POSCAR === #
        PC = AtomPosition(path + "/POSCAR")
        PC.read_POSCAR()
        cell = PC.latticeVectors

        # === read EIGENVAL === #
        EFM = EigenvalueFileManager(file_location=path + "/EIGENVAL", fermi=fermi, cell=cell)
        EFM.read_EIGENVAL()

    EFM.export_as_json(output_path+'/band_structure.json')

def generate_export_files(path:str, source:str=None, subfolders:bool=False, output_path:str=None, output_source:str=None, verbose:bool=False, bond_factor:float=None):
    """
    Export atomic position files from a specified source format to another format.

    This function is used to convert the format of atomic position files, which is often necessary for compatibility with different simulation tools or visualization software.

    Parameters:
    path (str): Path to the directory containing source format files.
    source (str, optional): Source format of the files (e.g., 'VASP').
    subfolders (bool, optional): Whether to include subfolders in the search (default False).
    output_path (str, optional): Directory path where the converted files will be saved.
    output_source (str, optional): Target format for exporting (e.g., 'PDB').
    verbose (bool, optional): If True, provides detailed output during execution.

    Returns:
    None
    """
    PT = Partition(path)
    PT.read_files( file_location=path, source=source, subfolders=subfolders, verbose=verbose)
    PT.export_files(file_location=output_path, source=output_source, label='enumerate', verbose=verbose)

def generate_plot(path:str, source:str=None, subfolders:bool=False, output_path:str=None, 
                    plot:str=None, verbose:bool=False, conteiner_index:int=None,
                    fermi:float=None, emin:float=None, emax:float=None,
                    cutoff:float=None, number_of_bins:int=None):
    """
    Generate plots from simulation data.

    This function processes simulation data and generates plots, such as band structure or molecular structures, based on the data and specified plot type.

    Parameters:
    path (str): Path to the directory containing the simulation data files.
    source (str, optional): Source of the files (e.g., 'VASP').
    subfolders (bool, optional): Whether to include subfolders in the search (default False).
    output_path (str, optional): Directory path where the plots will be saved.
    plot (str, optional): Type of plot to generate (e.g., 'band').
    verbose (bool, optional): If True, provides detailed output during execution.
    fermi (float, optional): Fermi level energy, important for certain types of plots.

    Returns:
    None
    """
    output_path = output_path if output_path is not None else '.'
    PT = Partition(path)
    PT.read_files( file_location=path, source=source, subfolders=subfolders, verbose=verbose)

    if plot.upper() == 'BANDS':
        if fermi is None:
            # === read DOCAR === #
            DM = DOSManager(path + "/DOSCAR")
            DM.read_DOSCAR()
            fermi = DM.fermi

        # === read EIGENVAL === #
        EFM = EigenvalueFileManager(file_location=path+"/EIGENVAL", fermi=fermi )
        EFM.read_EIGENVAL()

        EFM.plot(subtract_fermi=True, save=True, emin=emin, emax=emax)

    elif plot.upper() == 'RBF':
        if isinstance(conteiner_index,int): 
            PT.containers[conteiner_index].AtomPositionManager.plot_RBF(periodic_image=0, cutoff=cutoff, number_of_bins=number_of_bins, output_path=output_path,
                                                        bin_volume_normalize=True, number_of_atoms_normalize=True, density_normalize=True)
        else:
            for conteiner_index, conteiner in enumerate(PT.containers):
                conteiner.AtomPositionManager.plot_RBF(periodic_image=0, cutoff=cutoff, number_of_bins=number_of_bins, output_path=output_path,
                                                        bin_volume_normalize=True, number_of_atoms_normalize=True, density_normalize=True)


def generate_edit_positions(path: str, source: str = None, subfolders: bool = False, output_path: str = None, 
                            output_source: str = None, verbose: bool = False, edit: str = None, N: int = None, 
                            std: float = None, repeat: list = None, compress_min: list = None, compress_max: list = None):
    """
    Modifies and exports atomic positions based on specified editing operations.

    This function applies various editing operations like 'rattle', 'supercell', or 'compress' to the atomic positions
    in the provided path and exports the modified structures.

    Parameters:
        path (str): Path to the input files.
        source (str, optional): Source type of the input files.
        subfolders (bool, optional): Flag to include subfolders in file reading.
        output_path (str, optional): Path for exporting the edited files.
        output_source (str, optional): Source type for exporting files.
        verbose (bool, optional): Enables verbose output.
        edit (str, optional): Type of editing operation ('rattle', 'supercell', 'compress').
        N (int, optional): Parameter specific to the 'rattle' and 'compress' edit.
        std (float, optional): Standard deviation parameter for the 'rattle' edit.
        repeat (list, optional): Repetition vector for the 'supercell' edit.
        compress_min (list, optional): Minimum compression factor for the 'compress' edit.
        compress_max (list, optional): Maximum compression factor for the 'compress' edit.
    
    Raises:
        ValueError: If required parameters for specific edits are not provided.
    """
    # Initialize the DFTPartition object
    PT = Partition(path)

    # Read files from the specified path
    PT.read_files(file_location=path, source=source, subfolders=subfolders, verbose=verbose)

    # Apply the specified movement to the atomic positions
    if edit.lower() == 'rattle':
        # Ensure that N and std are provided for the 'rattle' move
        if N is None or std is None:
            raise ValueError("For the 'rattle' edit, both 'N' and 'std' parameters must be provided.")
        
        PT.generate_variants('rattle', [{'N': N, 'std': [std]}])
        PT.export_files(file_location=output_path, source=output_source, label='enumerate', verbose=verbose)

    if edit.lower() == 'supercell':
        # Ensure that N and std are provided for the 'rattle' move
        if repeat is None:
            raise ValueError("For the 'supercell' edit, the 'repeat' parameter must be provided.") 

        for container in PT.containers:
            container.AtomPositionManager.generate_supercell(repeat=repeat)
            name  = '_'.join( [ str(r) for r in repeat ] )
            container.file_location += f'/{name}'
        PT.export_files(file_location=output_path, source=output_source, label='enumerate', verbose=verbose)

    if edit.lower() == 'compress':
        #
        if compress_min is None and compress_max is None:
            raise ValueError("For the 'compress' edit, the 'compress_factor' parameter must be provided.") 

        PT.generate_variants(parameter='compress', values={'N': N, 'compress_min': compress_min, 'compress_max': compress_max} )
        PT.export_files(file_location=output_path, source=output_source, label='enumerate', verbose=verbose)

def generate_edit_configuration(path:str, source:str=None, subfolders:bool=False, output_path:str=None, output_source:str=None, verbose:bool=False, edit:str=None,
                            atom_ID:str=None, new_atom_ID:str=None ):
    """
    Edits the configuration of atomic structures by changing atomic IDs and exports the modified structures.

    This function changes the IDs of atoms in the provided structures based on the specified edit operations and 
    then exports these modified structures to the defined output path.

    Parameters:
        path (str): Path to the input files.
        source (str, optional): Source type of the input files.
        subfolders (bool, optional): Flag to include subfolders in file reading.
        output_path (str, optional): Path for exporting the edited files.
        output_source (str, optional): Source type for exporting files.
        verbose (bool, optional): Enables verbose output.
        edit (str, optional): Type of editing operation ('ATOM_ID').
        atom_ID (str, optional): Original atom ID to be changed.
        new_atom_ID (str, optional): New atom ID to replace the original.

    Raises:
        ValueError: If required parameters for specific edits are not provided.
    """

    # Initialize the DFTPartition object
    PT = Partition(path)

    # Read files based on provided parameters
    PT.read_files( file_location=path, source=source, subfolders=subfolders, verbose=verbose)
    print(PT.containers)
    if edit.upper() == 'ATOM_ID':
        PT.generate_variants(parameter='CHANGE_ATOM_ID', values={'atom_ID':atom_ID,'new_atom_ID':new_atom_ID}, )

    # Export the edited files
    PT.export_files(file_location=output_path, source=output_source, label='enumerate', verbose=verbose)

def generate_filter(path:str, source:str=None, subfolders:bool=False, output_path:str=None, output_source:str=None, verbose:bool=False, 
                    filter_class:str=None, container_property:str=None, filter_type:str=None, value:float=None, ID:str=None, traj:bool=False, N:int=None):
    """
    Filters atomic structures based on specified criteria and exports the filtered structures.

    This function applies filtering operations to atomic structures based on various properties like class,
    type, value, etc., and then exports these filtered structures to the defined output path.

    Parameters:
        path (str): Path to the input files.
        source (str, optional): Source type of the input files.
        subfolders (bool, optional): Flag to include subfolders in file reading.
        output_path (str, optional): Path for exporting the filtered files.
        output_source (str, optional): Source type for exporting files.
        verbose (bool, optional): Enables verbose output.
        filter_class (str, optional): Class of filter to apply.
        container_property (str, optional): Property of the container to filter by.
        filter_type (str, optional): Type of filter to apply.
        value (float, optional): Value threshold for filtering.
        ID (str, optional): Specific ID to filter by.
        traj (bool, optional): Flag to indicate trajectory filtering.
        N (int, optional): Number of structures to select.

    Raises:
        ValueError: If required parameters for specific filters are not provided.
    """
    # Initialize the DFTPartition object
    PT = Partition(path)

    # Read files and apply filters
    PT.read_files(file_location=path, source=source, subfolders=subfolders, verbose=verbose)
    PT.filter_conteiners(filter_class=filter_class, container_property=container_property, filter_type=filter_type, 
                        value=value, traj=traj, selection_number=N, ID=ID)
    # Export the filtered files
    PT.export_files(file_location=output_path, source=source, label='enumerate', verbose=verbose)

def generate_supercell_embedding(
                        unitcell_path:str,              unitcell_source:str, 
                        relax_supercell_path:str,       relax_supercell_source:str, 
                        unrelax_supercell_path:str,     unrelax_supercell_source:str, 
                        output_path:str=None,           output_source:str=None, verbose:bool=False, ):
    """
    Generates a supercell embedding from given unitcell and supercell structures and exports the result.

    This function reads the unitcell and both relaxed and unrelaxed supercell structures, generates a supercell
    embedding by combining these structures, and then exports the embedded supercell.

    Parameters:
        unitcell_path (str): Path to the unitcell file.
        unitcell_source (str): Source type of the unitcell file.
        relax_supercell_path (str): Path to the relaxed supercell file.
        relax_supercell_source (str): Source type of the relaxed supercell file.
        unrelax_supercell_path (str): Path to the unrelaxed supercell file.
        unrelax_supercell_source (str): Source type of the unrelaxed supercell file.
        output_path (str, optional): Path for exporting the embedded supercell.
        output_source (str, optional): Source type for exporting the supercell.
        verbose (bool, optional): Enables verbose output.

    Raises:
        ValueError: If required parameters for embedding are not provided.
    """
    # Initialize the DFTPartition object
    PT = Partition(relax_supercell_path)

    # Read unitcell and supercell structures
    PT.read_unitcell(file_location=unitcell_path, source=unitcell_source)

    PT.read_defect_supercell_unrelax(file_location=unrelax_supercell_path, source=unrelax_supercell_source )
    PT.read_defect_supercell_relax(file_location=relax_supercell_path, source=relax_supercell_source )

   # Generate supercell embedding
    PT.make_supercell_embedding()

    # Export the embedded supercell
    PT.export_defect_supercell_unrelax_Np1(source=output_source, file_location=output_path)

def add_arguments(parser):
    """
    Adds common arguments to the given subparser. This includes arguments for file paths, 
    verbosity, and other common settings used across various sub-commands.
    """
    parser.add_argument('--path', type=str, default='.', help='Path to the files directory')
    parser.add_argument('--source', type=str, choices=['VASP', 'OUTCAR', 'xyz', 'traj', 'cif', 'AIMS'], default='VASP', help='Source of calculation from which the files originate: VASP, molecular_dynamics, or force_field (default: VASP)')
    
    parser.add_argument('--output_path', type=str, default='.', help='Path for exporting VASP partition and scripts')
    parser.add_argument('--output_source', type=str, choices=['VASP', 'POSCAR','xyz', 'PDB', 'AIMS'], default='VASP', help='Source for exporting partition and scripts')

    parser.add_argument('--verbose', action='store_true', help='Display additional information')
    parser.add_argument('--subfolders', default=False, action='store_true', help='Read from all subfolders under the specified path')    
    parser.add_argument('--conteiner_index', type=int, help='')    


def main():
    """
    Main function to handle command line arguments and execute respective functions.
    This tool is designed for theoretical calculations in quantum mechanics and molecular dynamics,
    providing various sub-commands for different types of calculations and manipulations.
    """

    # Set up the main argument parser
    parser = argparse.ArgumentParser(description='Tool for theoretical calculations in quantum mechanics and molecular dynamics.')
    subparsers = parser.add_subparsers(dest='command', help='Available sub-commands')

    # =========== Sub-command to generate vacancy directory ===========
    parser_vacancy = subparsers.add_parser('vacancy', help='Generate configurations for vacancies in materials.')
    add_arguments(parser_vacancy)

    # =========== Sub-command to generate configurations for disassembling the surface ===========
    parser_disassemble = subparsers.add_parser('disassemble', help='Generate configurations for disassembling surfaces or layers in a material.')
    add_arguments(parser_disassemble)
    parser_disassemble.add_argument('--steps', type=int, default=5, help='Number of steps in the disassembly process (default: 5)')
    parser_disassemble.add_argument('--final_distance', type=float, default=5.0, help='Final distance between layers or atoms after disassembly (default: 5.0)')
    parser_disassemble.add_argument('--atoms_to_remove', type=int, help='Number of atoms to remove during disassembly.')

    # =========== Sub-command: dimer ===========
    parser_dimer = subparsers.add_parser('dimer', help='Search for optimal dimer configurations in a material.')
    add_arguments(parser_dimer)
    parser_dimer.add_argument('--labels', nargs='+', help='List of atom labels to include in the dimer search')
    parser_dimer.add_argument('--steps', type=int, default=10, help='Number of iterative steps in the dimer search (default: 10)')
    parser_dimer.add_argument('--vacuum', type=int, default=18, help='Vacuum distance around the dimer structure (default: 18A)')

    # =========== Sub-comando para generar script ===========
    parser_config = subparsers.add_parser('config', help='Read Position data from "path", read Configurtion data from "config_path" and export to "output_path".')
    add_arguments(parser_config)
    parser_config.add_argument('--config_path', type=str, required=True, help='')
    parser_config.add_argument('--config_source', type=str, required=True, help='')

    # =========== Sub-comando para generar BAND files ===========
    parser_bands = subparsers.add_parser('bands', help='Configure parameters for generating band calculation files from VASP data.')
    add_arguments(parser_bands)
    parser_bands.add_argument('--points', type=int, help='Specifies the number of k-points in each segment of the band path. It should be an integer value representing the total number of k-points along the path.')
    parser_bands.add_argument('--special_points', type=str, required=True, default='GMMLLXXG', help='Defines special points in the Brillouin zone for band calculations. Should be a character string representing points, for example, "GMXLG", indicating the high-symmetry points along the band path.')

    # =========== Sub-command for ganerate .JSON files from EIGENVAL ===========
    parser_bands2json = subparsers.add_parser('bands2json', help='Configure parameters for generating band calculation files from VASP data.')
    add_arguments(parser_bands2json)
    parser_bands2json.add_argument('--fermi', type=float, help='Specifies the energy of the fermi level.')

    # =========== Sub-command for export files from SOURCE format to OUTPUT format ===========
    parser_export_position = subparsers.add_parser('export', help='Export atomic positions from a specified source format to a desired output format. This is useful for converting file formats for compatibility with various simulation and visualization tools.')
    add_arguments(parser_export_position)
    parser_export_position.add_argument('--bond_factor', type=float, default=1.1, required=False, help='')

    # =========== Sub-command for PLOT files from SOURCE format to OUTPUT format ===========
    parser_plot = subparsers.add_parser('plot', help='Generates plots based on data from a specified source. This can include plotting energy bands, density of states, or molecular structures, depending on the input data and specified plot type.')
    add_arguments(parser_plot)
    parser_plot.add_argument('--plot', type=str, choices=['bands', 'RBF'], help='Specifies the type of plot to generate, such as "band" for band structure plots.')
    parser_plot.add_argument('--fermi', type=float, help='Specifies the energy of the Fermi level, which is essential for accurate band structure plots.')
    parser_plot.add_argument('--emin', type=float, help='')
    parser_plot.add_argument('--emax', type=float, help='')
    parser_plot.add_argument('--cutoff', type=float, default=6.0, help='')
    parser_plot.add_argument('--number_of_bins', type=int, default=100, help='')

    # ========== Sub-command: edit_positions ===========
    parser_edit = subparsers.add_parser('edit_positions', help='Modify atomic positions in the input files, allowing operations like "rattling" to introduce small random displacements.')
    add_arguments(parser_edit)
    parser_edit.add_argument('--edit', type=str, default='rattle', help='Type of modification to apply to atomic positions. E.g., "rattle" for random displacements.')
    parser_edit.add_argument('--std', type=float, required=False, help='Standard deviation for displacement distribution in "rattle" operation, defining displacement magnitude.')
    parser_edit.add_argument('--N', type=int, required=False, help='Number of applications of the selected operation. Defaults to 1 if not specified.')
    parser_edit.add_argument('--repeat', type=int, nargs=3, default=[1, 1, 1], help='Repeat the unit cell in x, y, z dimensions respectively. Format: x y z')
    parser_edit.add_argument('--compress_min', type=float, nargs=3, default=[1, 1, 1], help='Minimum compression factors in x, y, z for the "compress" operation. Format: x y z')
    parser_edit.add_argument('--compress_max', type=float, nargs=3, default=[1, 1, 1], help='Maximum compression factors in x, y, z for the "compress" operation. Format: x y z')

    # ========== Sub-command: edit_configuration ===========
    parser_edit_config = subparsers.add_parser('edit_configuration', help='Modify the configuration settings of your simulation or calculation process.')
    add_arguments(parser_edit_config)
    parser_edit_config.add_argument('--edit', type=str, help='Specify the type of configuration modification.')
    parser_edit_config.add_argument('--ID', type=str, required=False, help='Identifier for the configuration element to modify.')
    parser_edit_config.add_argument('--new_ID', type=str, required=False, help='New identifier to assign to the configuration element.')

    # ========== Sub-command: filter ===========
    parser_filter = subparsers.add_parser('filter', help='Apply filters to select specific data based on various criteria.')
    add_arguments(parser_filter)
    parser_filter.add_argument('--filter', choices=['random', 'flat_histogram', 'binary'], type=str, help='Type of filter to apply: random, flat_histogram, or binary.')
    parser_filter.add_argument('--property', choices=['E', 'forces', 'has_ID', 'has_not_ID'], type=str, help='Property to use for filtering: Energy (E), forces, or atom IDs.')
    parser_filter.add_argument('--type', choices=['max', 'min'], type=str, help='Filter type: max (maximum value) or min (minimum value).')
    parser_filter.add_argument('--value', type=float, required=False, help='Value to compare against for the filter.')
    parser_filter.add_argument('--ID', type=str, required=False, help='Atom ID for filtering, if applicable.')
    parser_filter.add_argument('--traj', default=False, action='store_true', help='Enable trajectory-based filtering.')
    parser_filter.add_argument('--N', default=1, type=float, help='Number of elements to select when filtering.')

    # ========== Sub-command: supercell_embedding ===========
    parser_supercell_embedding = subparsers.add_parser('supercell_embedding', help='Embed a unit cell into a supercell for large-scale calculations or visualizations.')
    add_arguments(parser_supercell_embedding)
    parser_supercell_embedding.add_argument('--unitcell_path', type=str, default='.', help='Path to the unit cell file. (default: .)')
    parser_supercell_embedding.add_argument('--unitcell_source', type=str, choices=['VASP', 'OUTCAR', 'xyz', 'traj', 'cif', 'AIMS'], default='VASP', help='Source format of the unit cell file: VASP, OUTCAR, xyz, etc. (default: VASP)')
    parser_supercell_embedding.add_argument('--notrelax_supercell_path', type=str, default='.', help='Path to the non-relaxed supercell file.')
    parser_supercell_embedding.add_argument('--notrelax_supercell_source', type=str, choices=['VASP', 'OUTCAR', 'xyz', 'traj', 'cif', 'AIMS'], default='VASP', help='Source format of the non-relaxed supercell file.')
    
    args = parser.parse_args()

    # Handle execution based on the specified sub-command
    if args.command == 'vacancy':
        generate_vacancy(args.path, subfolders=args.subfolders, verbose=args.verbose)
    
    elif args.command == 'disassemble':
        generate_disassemble_surface(args.path, steps=args.steps, final_distance=args.final_distance, atoms_to_remove=args.atoms_to_remove, subfolders=args.subfolders, verbose=args.verbose)
    
    elif args.command == 'dimer':
        generate_dimers(path=args.path, labels=args.labels, steps=args.steps, vacuum=args.vacuum, subfolders=args.subfolders, verbose=args.verbose)
    
    elif args.command == 'config':
        generate_config(path=args.path, source=args.source, subfolders=args.subfolders, 
                        config_path=args.config_path, config_source=args.config_source,
                        output_path=args.output_path, output_source=args.output_source, verbose=args.verbose)
    
    elif args.command == 'bands':
        generate_band_calculation(path=args.path, source=args.source, points=args.points, special_points=args.special_points, 
                        subfolders=args.subfolders, verbose=args.verbose, output_path=args.output_path)
    
    elif args.command == 'bands2json':
        generate_json_from_bands(path=args.path, source=args.source, fermi=args.fermi,
                        subfolders=args.subfolders, verbose=args.verbose, output_path=args.output_path)

    elif args.command == 'export':
        generate_export_files(path=args.path, source=args.source, subfolders=args.subfolders, 
            verbose=args.verbose, output_path=args.output_path, output_source=args.output_source, bond_factor=args.bond_factor)

    elif args.command == 'plot':
        generate_plot(path=args.path, source=args.source, subfolders=args.subfolders, conteiner_index=args.conteiner_index,
            verbose=args.verbose, output_path=args.output_path, plot=args.plot, fermi=args.fermi,
            emin=args.emin, emax=args.emax, cutoff=args.cutoff, number_of_bins=args.number_of_bins)

    elif args.command == 'edit_positions':
        generate_edit_positions(path=args.path, source=args.source, subfolders=args.subfolders, verbose=args.verbose, 
            output_path=args.output_path, output_source=args.output_source, 
            edit=args.edit, std=args.std, N=args.N, repeat=args.repeat, compress_min=args.compress_min, compress_max=args.compress_max )

    elif args.command == 'edit_configuration':
        generate_edit_configuration(path=args.path, source=args.source, subfolders=args.subfolders, verbose=args.verbose, 
            output_path=args.output_path, output_source=args.output_source, 
            edit=args.edit, atom_ID=args.ID, new_atom_ID=args.new_ID )

    elif args.command == 'filter':
        generate_filter(path=args.path, source=args.source, subfolders=args.subfolders, verbose=args.verbose, output_path=args.output_path, output_source=args.output_source,
                filter_class=args.filter, container_property=args.property, filter_type=args.type, value=args.value, traj=args.traj, N=args.N, ID=args.ID)

    elif args.command == 'supercell_embedding':
        generate_supercell_embedding(
                        relax_supercell_path=args.path,     relax_supercell_source=args.source, 
                        unrelax_supercell_path=args.notrelax_supercell_path,   unrelax_supercell_source=args.notrelax_supercell_source, 
                        unitcell_path=args.unitcell_path,            unitcell_source=args.unitcell_source, 
                        output_path=args.output_path,       output_source=args.output_source,
                        verbose=args.verbose )

if __name__ == '__main__':
    main()