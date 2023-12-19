try:
    from sage_lib.partition.PartitionManager import PartitionManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing PartitionManager: {str(e)}\n")
    del sys

try:
    import numpy as np
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing numpy: {str(e)}\n")
    del sys

class Config_builder(PartitionManager):
    def __init__(self, file_location:str=None, name:str=None, **kwargs):
        super().__init__(name=name, file_location=file_location)
        
    def handleKPoints(self, container:object, values:list, container_index:int,  file_location:str=None):
        """
        Handles the configuration for KPoints.

        This method creates container copies for each set of k-points values, updates the 
        subdivisions, and generates execution scripts for each container.

        Parameters:
        container (object): The container to be copied and updated.
        values (list): List of k-points values.
        container_index (int): Index of the container.
        file_location (str): File location for the container copy.

        Returns:
        list: A list of container copies with updated k-points configurations.
        """
        sub_directories, containers = [], []

        for v in values:
            # Copy and update container for each set of k-point values
            container_copy = self.copy_and_update_container(container, f'/KPOINTConvergence/{v[0]}_{v[1]}_{v[2]}', file_location)
            container_copy.KPointsManager.subdivisions = [v[0], v[1], v[2]]
            sub_directories.append(f'{v[0]}_{v[1]}_{v[2]}')
            containers.append(container_copy)

        # Generate execution script for each updated container
        self.generate_execution_script_for_each_container(sub_directories, container.file_location + '/KPOINTConvergence')
        return containers

    def handleInputFile(self, container:object, values:list, container_index:int, file_location:str=None):
        """
        Handles the configuration for input files.

        This method updates the parameters of the input file for each value in the provided list
        and generates execution scripts for each container.

        Parameters:
        container (object): The container to be copied and updated.
        values (list): List of parameter values for input file configuration.
        container_index (int): Index of the container.
        file_location (str): File location for the container copy.

        Returns:
        list: A list of container copies with updated input file configurations.
        """
        sub_directories, containers = [], []

        for v in values:
            # Copy and update container for each parameter value
            container_copy = self.copy_and_update_container(container, f'/{parameter}_analysis/{v}', file_location)
            container_copy.InputFileManager.parameters[parameter.upper()] = ' '.join(v) if v is list else v 
            sub_directories.append('_'.join(map(str, v)) if isinstance(v, list) else str(v))
            containers.append(container_copy)

        # Generate execution script for each updated container
        self.generate_execution_script_for_each_container(sub_directories, container.file_location + f'/{parameter}_analysis')
        return containers

    def handleAtomIDChange(self, container:object, values:dict, container_index:int, file_location:str=None):
        """
        Handles the configuration for changing atom IDs.

        This method updates the atom ID in the container based on the provided values.

        Parameters:
        container (object): The container to be copied and updated.
        values (dict): Dictionary containing the old and new atom IDs.
        container_index (int): Index of the container.
        file_location (str): File location for the container copy.

        Returns:
        list: A list containing the container copy with updated atom IDs.
        """
        sub_directories, containers = [], []

        atom_ID, new_atom_ID = values['atom_ID'], values['new_atom_ID']
        container_copy = self.copy_and_update_container(container, f'/AtomIDChange_{atom_ID}-{new_atom_ID}', file_location)
        container_copy.AtomPositionManager.change_ID(atom_ID=values['atom_ID'], new_atom_ID=values['new_atom_ID'])

        # Copy and update container for atom ID change
        containers.append(container_copy)

        return containers


