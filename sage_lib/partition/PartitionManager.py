try:
    from sage_lib.master.FileManager import FileManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing FileManager: {str(e)}\n")
    del sys

try:
    from sage_lib.single_run.SingleRun import SingleRun
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing SingleRun: {str(e)}\n")
    del sys

try:
    from sage_lib.output.OutFileManager import OutFileManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing OutFileManager: {str(e)}\n")
    del sys

try:
    from sage_lib.input.structure_handling_tools.AtomPosition import AtomPosition
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing AtomPosition: {str(e)}\n")
    del sys

try:
    import numpy as np
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing numpy: {str(e)}\n")
    del sys

try:
    import os 
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing os: {str(e)}\n")
    del sys

try:
    import copy
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing copy: {str(e)}\n")
    del sys

try:
    import re
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing re: {str(e)}\n")
    del sys

try:
    from ase.io import Trajectory
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing ase.io.Trajectory: {str(e)}\n")
    del sys


class PartitionManager(FileManager): 
    """
    PartitionManager class for managing and partitioning simulation data.

    Inherits:
    - FileManager: For file management functionalities.

    Attributes:
    - file_location (str): File path for data files.
    - containers (list): Containers to hold various data structures.
    """
    def __init__(self, file_location:str=None, name:str=None, **kwargs):
        """
        Initializes the PartitionManager object.

        Args:
        - file_location (str, optional): File path location.
        - name (str, optional): Name of the partition.
        - **kwargs: Additional arguments.
        """
        super().__init__(name=name, file_location=file_location)
        self._containers = []
        self._time = []
        self._N = None

    @property
    def N(self):
        if self._containers is None:
            return 0
        elif self._containers is list:
            return len(self._containers)
        elif self._containers is np.array:
            return self._containers.shape[0]
        else:
            return None

    def add_container(self, container: object):
        """
        Add a new container to the list of containers.

        Parameters:
            container (object): The container object to be added.
        """
        self.containers.append(container)

    def remove_container(self, container: object):
        """
        Remove a container from the list of containers.

        Parameters:
            container (object): The container object to be removed.
        """
        self.containers.remove(container)

    def empty_container(self):
        """
        Empty the list of containers.
        """
        self.containers = []

    def apply_filter_mask(self, mask:list) -> bool:
        """

        """
        self._containers = [conteiner for conteiner, m in zip(self.containers, mask) if m == 1]

    def read_Config_Setup(self, file_location:str=None, source='VASP', verbose=False):
        '''
        '''
        file_location = file_location if type(file_location) == str else self.file_location

        if source.upper() == 'VASP':
            SR = self.readVASPFolder(file_location=file_location, add_container=False, verbose=verbose)
            if SR.AtomPositionManager is not None:
                SR.InputFileManager.set_LDAU( SR.AtomPositionManager.uniqueAtomLabels )

        for c_i, container in enumerate(self.containers):
            container.InputFileManager = SR.InputFileManager
            container.KPointsManager = SR.KPointsManager
            container.PotentialManager = SR.PotentialManager
            container.BashScriptManager = SR.BashScriptManager
            container.vdw_kernel_Handler = SR.vdw_kernel_Handler
            container.WaveFileManager = SR.WaveFileManager
            container.ChargeFileManager = SR.ChargeFileManager

    def read_files(self, file_location:str=None, source:str='VASP', subfolders:bool=False, verbose:bool=False):
        '''
        '''
        if subfolders:
            self.readSubFolder(file_location=file_location, source=source, verbose=verbose)

        else:
            if source.upper() == 'VASP':
                self.readVASPFolder(file_location=file_location, add_container=True, verbose=verbose)
            elif source.upper() == 'TRAJ':
                self.read_traj(file_location=file_location, add_container=True, verbose=verbose)
            elif source.upper() == 'XYZ':
                self.read_XYZ(file_location=file_location, add_container=True, verbose=verbose)
            elif source.upper() == 'OUTCAR':
                self.read_OUTCAR(file_location=file_location, add_container=True, verbose=verbose)
    
    def readSubFolder(self, file_location:str=None, source:str='VASP', verbose=False):
        """
        Reads files from a specified directory and its subdirectories.

        This function is designed to traverse through a directory (and its subdirectories) to read files 
        according to the specified source type. It handles various file-related errors gracefully, providing 
        detailed information if verbose mode is enabled.

        Args:
            file_location (str, optional): The root directory from where the file reading starts. 
                                           Defaults to the instance's file_location attribute if not specified.
            source (str): Type of the source files to be read (e.g., 'OUTCAR' for VASP output files).
            verbose (bool, optional): If True, enables verbose output including error traces.
        """
        file_location = file_location if type(file_location) == str else self.file_location
        for root, dirs, files in os.walk(file_location):
            if verbose: print(root, dirs, files)

            if source == 'OUTCAR': file_location_edited = f'{root}/OUTCAR'
            else: file_location_edited = f'{root}' 

            try:
                SR = self.read_files(file_location=file_location_edited, source=source, subfolders=False, verbose=verbose)
            except FileNotFoundError:
                self._handle_error(f"File not found at {file_location_edited}", verbose)
            except IOError:
                self._handle_error(f"IO error reading file at {file_location_edited}", verbose)
            except Exception as e:
                self._handle_error(f"Unexpected error: {e}", verbose)

    def read_traj(self, file_location:str=None, add_container:bool=True, verbose=False):
        """
        Reads a trajectory file and stores each frame along with its time information.

        Args:
            file_location (str, optional): The file path of the trajectory file.
            verbose (bool, optional): If True, enables verbose output.

        Notes:
            This method updates the containers with SingleRun objects representing each frame.
            If available, time information is also stored.
        """
        file_location = file_location if type(file_location) == str else self.file_location
        from ase.io import Trajectory
        traj = Trajectory(file_location)

        for atoms in traj:
            SR = SingleRun(file_location)
            SR.read_ASE(ase_atoms=atoms) 
            if add_container and SR.AtomPositionManager is not None: 
                # Store the frame
                self.add_container(container=SR)
                # Store the time information if it's available
                if hasattr(atoms, 'get_time'):
                    self._time.append(atoms.get_time())

        del Trajectory


    def read_XYZ(self, file_location:str=None, add_container:bool=True, verbose:bool=False):
        '''
        '''
        file_location = file_location if type(file_location) == str else self.file_location

        lines =list(self.read_file(file_location,strip=False))
        container = []

        for i, line in enumerate(lines):
            if line.strip().isdigit():
                num_atoms = int(line.strip())
                if num_atoms > 0:
                    SR = SingleRun(file_location)
                    SR.AtomPositionManager = AtomPosition()
                    SR.AtomPositionManager.read_XYZ(lines=lines[i:i+num_atoms+2])

                    container.append(SR)

                    if add_container and SR.AtomPositionManager is not None: 
                            self.add_container(container=SR)
        return container

    def read_OUTCAR(self, file_location:str=None, add_container:bool=True,verbose=False):
        '''
        '''
        OF = OutFileManager(file_location)
        OF.readOUTCAR()

        for APM in OF.AtomPositionManager:
            SR = SingleRun(file_location)
            SR._AtomPositionManager = APM
            SR._InputFileManager = OF.InputFileManager
            SR._KPointsManager = OF._KPointsManager
            SR._PotentialManager = OF._PotentialManager
            if add_container and SR.AtomPositionManager is not None: 
                self.add_container(container=SR)

    def readVASPFolder(self, file_location:str=None, add_container:bool=True, verbose:bool=False):
        '''
        '''
        file_location = file_location if type(file_location) == str else self.file_location
        SR = SingleRun(file_location)
        SR.readVASPDirectory()        
        if add_container and SR.AtomPositionManager is not None: 
            self.add_container(container=SR)

        return SR

    def export_files(self, file_location:str=None, source:str='VASP', label:str=None, bond_factor:float=None, verbose:bool=False):
        """
        Exports files for each container in a specified format.

        The function iterates over all containers and exports them according to the specified format.
        In case of an error during export, it logs the error (if verbose is True) and continues with the next container.

        Args:
            file_location (str): The base directory for exporting files.
            source (str): The format to export files in ('VASP', 'POSCAR', 'XYZ', 'PDB', 'ASE').
            label (str): Labeling strategy for exported files ('enumerate' or 'fixed').
            bond_factor (float): The bond factor to use for PDB export.
            verbose (bool): If True, enables verbose output including error messages.
        """
        label = label if isinstance(label, str) else 'fixed'
        file_locations = []

        for c_i, container in enumerate(self.containers):
            try:
                if label == 'enumerate':
                    file_location_edited = file_location + f'/{c_i:03d}'
                elif label == 'fixed':
                    file_location_edited = container.file_location

                if source.upper() != 'XYZ':
                    self.create_directories_for_path(file_location_edited)
                else:
                    self.create_directories_for_path(file_location)
                    
                # Export based on the specified source format
                if source.upper() == 'VASP':
                    container.exportVASP(file_location=file_location_edited)
                elif source.upper() == 'POSCAR':
                    container.AtomPositionManager.export_as_POSCAR(file_location=file_location_edited + '/POSCAR')
                elif source.upper() == 'XYZ':
                    container.AtomPositionManager.export_as_xyz(file_location=file_location + '/config.xyz', save_to_file='a')
                elif source.upper() == 'PDB':
                    container.AtomPositionManager.export_as_PDB(file_location=file_location_edited + '/structure.pdb', bond_factor=bond_factor)
                elif source.upper() == 'ASE':
                    container.AtomPositionManager.export_as_ASE(file_location=file_location_edited + '/ase.obj')

                file_locations.append(file_location_edited)

            except Exception as e:
                if verbose:
                    print(f"Failed to export container {c_i}: {e}")

        self.generate_execution_script_for_each_container(directories=file_locations, file_location='.')

    def export_configXYZ(self, file_location:str=None, verbose:bool=False):
        '''
        '''
        file_location  = file_location if file_location else self.file_location+'_config.xyz'
        with open(file_location, 'w'):pass # Create an empty file

        for container_index, container in enumerate(self.containers):
            if container.OutFileManager is not None:    
                container.OutFileManager.export_configXYZ(file_location=file_location, save_to_file='a', verbose=False)

        if verbose:
            print(f"XYZ content has been saved to {file_location}")

        return True
    
    def _is_redundant(self, containers:list=None, new_container:object=None):
        """
        Checks if a new container is redundant within existing containers.

        Args:
        - new_container (object): The new container to check.
        - containers (list, optional): List of existing containers.

        Returns:
        - bool: True if redundant, False otherwise.
        """
        containers = containers if containers is not None else self.containers
        return any(np.array_equal(conteiner.atomPositions, new_container.atomPositions) for conteiner in containers)

    def summary(self, ) -> str:
        """
        Generates a summary string of the PartitionManager's current state.

        Returns:
            str: A summary string detailing the file location and the number of containers managed.
        """
        text_str = ''
        text_str += f'{self.file_location}\n'
        text_str += f'> Conteiners : { len(self.containers) }\n'
        return text_str
    
    def copy_and_update_container(self, container, sub_directory: str, file_location=None):
        """
        Creates a deep copy of a given container and updates its file location.

        Args:
            container (object): The container object to be copied.
            sub_directory (str): The subdirectory to append to the container's file location.
            file_location (str, optional): Custom file location for the new container. If None, appends sub_directory to the original container's file location.

        Returns:
            object: The copied and updated container object.
        """
        container_copy = copy.deepcopy(container)
        container_copy.file_location = f'{container.file_location}{sub_directory}' if file_location is None else file_location
        return container_copy

    def generate_execution_script_for_each_container(self, directories: list = None, file_location: str = None):
        """
        Generates and writes an execution script for each container in the specified directories.

        Args:
            directories (list, optional): List of directory paths for which the execution script is to be generated.
            file_location (str, optional): The file path where the generated script will be saved.

        Notes:
            The script 'RUNscript.sh' will be generated and saved to each specified directory.
        """
        self.create_directories_for_path(file_location)
        script_content = self.generate_script_content('RUNscript.sh', directories)
        self.write_script_to_file(script_content, f"{file_location}/execution_script_for_each_container.py")

    def generate_master_script_for_all_containers(self, directories: list = None, file_location: str = None):
        """
        Generates a master script that includes execution scripts for all containers.

        Args:
            directories (list, optional): List of directory paths to include in the master script.
            file_location (str, optional): The file path where the master script will be saved.

        Notes:
            The script 'execution_script_for_each_container.py' will be generated and saved to the specified file location.
        """
        self.create_directories_for_path(file_location)
        script_content = self.generate_script_content('execution_script_for_each_container.py', directories)
        self.write_script_to_file(script_content, f"{file_location}/master_script_for_all_containers.py")

    def generate_script_content(self, script_name: str, directories: list = None) -> str:
        """
        Generates the content for a script that runs specified scripts in given directories.

        Args:
            script_name (str): The name of the script to run in each directory.
            directories (list, optional): A list of directories where the script will be executed.

        Returns:
            str: The generated script content as a string.
        """
        directories_str = "\n".join([f"    '{directory}'," for directory in directories])
        return f'''#!/usr/bin/env python3
import os
import subprocess

original_directory = os.getcwd()

directories = [
{directories_str}
]

for directory in directories:
    os.chdir(directory)
    subprocess.run(['chmod', '+x', '{script_name}'])
    subprocess.run(['sbatch', '{script_name}'])
    os.chdir(original_directory)
'''

    def write_script_to_file(self, script_content: str, file_path: str):
        """
        Writes the provided script content to a file at the specified path.

        Args:
            script_content (str): The content of the script to be written.
            file_path (str): The file path where the script will be saved.

        Notes:
            This method creates or overwrites the file at the specified path with the given script content.
        """
        with open(file_path, "w") as f:
            f.write(script_content)




'''
DP.exportVaspPartition()

print(DP.containers[0].AtomPositionManager.pbc)
DP.generateDFTVariants('band_structure', values=[{'points':20, 'special_points':'GMLCXG'}])
DP.exportVaspPartition()


path = '/home/akaris/Documents/code/Physics/VASP/v6.1/files/POSCAR/Cristals/test/files'
DP = DFTPartition(path)
DP.readVASPSubFolder(v=False)
DP.readConfigSetup('/home/akaris/Documents/code/Physics/VASP/v6.1/files/POSCAR/Cristals/test/config')
DP.generate_execution_script_for_each_container([ f'{n:03d}'for n, c in enumerate(DP.containers) ], '/home/akaris/Documents/code/Physics/VASP/v6.1/files/POSCAR/Cristals/test/calcs')

#DP.read_configXYZ()

'''




'''
DP.export_configXYZ()

path = '/home/akaris/Documents/code/Physics/VASP/v6.1/files/dataset/CoFeNiOOH_jingzhu/surf_CoFe_4H_4OH/MAG'

DP = DFTPartition(path)

DP.readVASPFolder(v=True)

DP.generateDFTVariants('Vacancy', [1], is_surface=True)
#DP.generateDFTVariants('KPOINTS', [[n,n,1] for n in range(1, 15)] ) 


path = '/home/akaris/Documents/code/Physics/VASP/v6.1/files/dataset/CoFeNiOOH_jingzhu/surf_CoFe_4H_4OH/MAG'
DP = DFTPartition(path)
DP.readVASPFolder(v=True)
DP.generateDFTVariants('NUPDOWN', [n for n in range(0, 10, 1)] )
DP.writePartition()

path = '/home/akaris/DocumeEENnts/code/Physics/VASP/v6.1/files/POSCAR/Cristals/NiOOH/*OH surface with Fe(HS)'
path = '/home/akaris/Documents/code/Physics/VASP/v6.1/files/POSCAR/Cristals/NiOOH/*OH surface for pure NiOOH'
#DP = DFTPartition('/home/akaris/Documents/code/Physics/VASP/v6.1/files/bulk_optimization/Pt/parametros/ENCUT_optimization_252525_FCC')
DP = DFTPartition(path)
#DP.readVASPSubFolder(v=True)
DP.readVASPFolder(v=True)

#DP.generateDFTVariants('Vacancy', [1], is_surface=True)
#DP.generateDFTVariants('KPOINTS', [[n,n,1] for n in range(1, 15)] )    
DP.generateDFTVariants('ENCUT', [n for n in range(200, 1100, 45)] )

DP.writePartition()

DP.generateDFTVariants('ENCUT', [ E for E in range(400,700,30)] )
print( DP.summary() )

print( DP.containers[0].AtomPositionManager.summary() )
'''

