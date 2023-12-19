try:
    from sage_lib.partition.partition_builder.BandStructure_builder import BandStructure_builder
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing BandStructure_builder: {str(e)}\n")
    del sys

try:
    from sage_lib.partition.partition_builder.Config_builder import Config_builder
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing Config_builder: {str(e)}\n")
    del sys

try:
    from sage_lib.partition.partition_builder.Crystal_builder import Crystal_builder
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing Crystal_builder: {str(e)}\n")
    del sys

try:
    from sage_lib.partition.partition_builder.CrystalDefect_builder import CrystalDefect_builder
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing CrystalDefect_builder: {str(e)}\n")
    del sys

try:
    from sage_lib.partition.partition_builder.ForceField_builder import ForceField_builder
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing ForceField_builder: {str(e)}\n")
    del sys

try:
    from sage_lib.partition.partition_builder.Molecule_builder import Molecule_builder
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing Molecule_builder: {str(e)}\n")
    del sys

try:
    from sage_lib.partition.partition_builder.PositionEditor_builder import PositionEditor_builder
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing PositionEditor_builder: {str(e)}\n")
    del sys

try:
    from sage_lib.partition.partition_builder.SurfaceStates_builder import SurfaceStates_builder
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing SurfaceStates_builder: {str(e)}\n")
    del sys

try:
    from sage_lib.partition.partition_builder.VacuumStates_builder import VacuumStates_builder
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing VacuumStates_builder: {str(e)}\n")
    del sys

try:
    from sage_lib.partition.partition_builder.Filter_builder import Filter_builder
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing Filter_builder: {str(e)}\n")
    del sys

try:
    from sage_lib.partition.partition_builder.SupercellEmbedding_builder import SupercellEmbedding_builder
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing SupercellEmbedding_builder: {str(e)}\n")
    del sys

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

class Partition(BandStructure_builder, Config_builder, Crystal_builder, CrystalDefect_builder, ForceField_builder, 
                 Molecule_builder, PositionEditor_builder, SurfaceStates_builder, VacuumStates_builder, Filter_builder, SupercellEmbedding_builder):
    """
    The Partition class is designed to handle various operations related to different types
    of crystal structure manipulations. It inherits from multiple builder classes, each
    specialized in a specific aspect of crystal structure and simulation setup.
    """

    def __init__(self, file_location:str=None, name:str=None, **kwargs):
        """
        Initializes the Partition class with the provided file location and name.

        Args:
            file_location (str, optional): The path to the file or directory where the data is stored.
            name (str, optional): The name associated with this instance of the Partition class.
            kwargs: Additional keyword arguments.
        """
        #super().__init__(name=name, file_location=file_location)

        # BandStructure_builder: Likely involved in constructing or analyzing the band structure of materials. 

        # This could include calculations related to the electronic properties of solids, such as band gaps, 
        # electronic density of states, etc.
        BandStructure_builder.__init__(self, name=name, file_location=file_location, **kwargs)
        # Config_builder: Potentially used for setting up or managing configuration settings. This could 
        # be related to simulation parameters, computational settings, or environment configurations.
        Config_builder.__init__(self, name=name, file_location=file_location, **kwargs)
        # Crystal_builder: Likely used for creating or manipulating crystal structures. This class might 
        # handle tasks like generating crystal lattices, defining unit cells, or applying crystallographic transformations.
        Crystal_builder.__init__(self, name=name, file_location=file_location, **kwargs)
        # CrystalDefect_builder: Probably focused on introducing and managing defects within crystal structures, 
        # such as vacancies, interstitials, or dislocations. Useful in studies of material properties influenced by imperfections.
        CrystalDefect_builder.__init__(self, name=name, file_location=file_location, **kwargs)
        # ForceField_builder: Likely used for defining or constructing force fields in molecular dynamics simulations. 
        # This could involve setting up parameters like bond strengths, angles, and torsional forces.
        ForceField_builder.__init__(self, name=name, file_location=file_location, **kwargs)
        # Molecule_builder: Presumably used for creating and manipulating molecular structures. This may include 
        # tasks like building molecular models, adding or removing atoms or groups, and setting molecular geometries.
        Molecule_builder.__init__(self, name=name, file_location=file_location, **kwargs)
        # PositionEditor_builder: Probably designed for editing and manipulating atomic positions. This could 
        # involve tasks like translating, rotating, or scaling atomic structures.
        PositionEditor_builder.__init__(self, name=name, file_location=file_location, **kwargs)
        # SurfaceStates_builder: Likely focuses on the properties and states of material surfaces. This might include 
        # surface energy calculations, adsorption studies, or surface reconstruction analyses.
        SurfaceStates_builder.__init__(self, name=name, file_location=file_location, **kwargs)
        # VacuumStates_builder: Potentially involved in simulating or analyzing states in a vacuum environment. 
        # This could be relevant in studies of isolated molecules or atoms, free from interactions with a surrounding medium.
        VacuumStates_builder.__init__(self, name=name, file_location=file_location, **kwargs)
        # Filter_builder: Possibly used for creating filters or criteria for selecting specific atoms, molecules, 
        # or structures based on certain properties or conditions.
        Filter_builder.__init__(self, name=name, file_location=file_location, **kwargs)
        # SupercellEmbedding_builder: Likely used in the context of supercell models in crystallography or materials science. 
        # This class might handle the creation or manipulation of supercells for periodic boundary condition simulations.
        SupercellEmbedding_builder.__init__(self, name=name, file_location=file_location, **kwargs)

    def generate_variants(self, parameter: str, values:np.array=None, file_location: str = None) -> bool:
        """
        Generates variants of the current container set based on the specified parameter and its range of values.

        This method iterates over the existing containers and applies different modifications
        according to the specified parameter (e.g., KPOINTS, VACANCY). The result is a new set
        of containers with the applied variations.

        Args:
            parameter (str): The parameter based on which the variants are to be generated.
            values (np.array, optional): The range of values to be applied for the parameter.
            file_location (str, optional): The location where the generated data should be stored.

        Returns:
            bool: True if the operation is successful, False otherwise.
        """
        containers = []
        directories = ['' for _ in self.containers]
        parameter = parameter.upper().strip()
        for container_index, container in enumerate(self.containers):

            if parameter.upper() == 'KPOINTS':
                containers += self.handleKPoints(container, values, container_index,  file_location) 
                directories[container_index] = 'KPOINTConvergence'

            elif container.InputFileManager and parameter.upper() in container.InputFileManager.parameters_data:
                containers += self.handleInputFile(container, values, parameter,  container_index, file_location)
                directories[container_index] = f'{parameter}_analysis'

            elif parameter.upper() == 'VACANCY':
                containers += self.handleVacancy(container, values, container_index, file_location)
                directories[container_index] = 'Vacancy'

            elif parameter.upper() == 'BAND_STRUCTURE':
                containers += self.handleBandStruture(container, values, container_index, file_location)
                directories[container_index] = 'band_structure'

            elif parameter.upper() == 'RATTLE':
                containers += self.handleRattle(container, values, container_index, file_location)
                directories[container_index] = 'rattle'

            elif parameter.upper() == 'COMPRESS':
                containers += self.handleCompress(container, values, container_index, file_location)
                directories[container_index] = 'compress'

            elif parameter.upper() == 'CHANGE_ATOM_ID':
                containers += self.handleAtomIDChange(container, values, container_index, file_location)
                directories[container_index] = 'compress'

        self.containers = containers
        #self.generate_master_script_for_all_containers(directories, file_location if not file_location is None else container.file_location )

'''
path = '/home/akaris/Documents/code/Physics/VASP/v6.2/files/dataset/CoFeNiOOH_jingzhu/bulk_NiFe/rattle'

DP.readVASPFolder(v=False)
DP.generate_variants('rattle', [ {'N':20, 'std':[0.13]} ] )
'''
