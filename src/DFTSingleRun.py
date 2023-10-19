try:
    KPointsManager = __import__('KPointsManager').KPointsManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing KPointsManager: {str(e)}\n")
    del sys

try:
    InputDFT = __import__('InputDFT').InputDFT
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing InputDFT: {str(e)}\n")
    del sys

try:
    PotentialManager = __import__('PotentialManager').PotentialManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing PotentialManager: {str(e)}\n")
    del sys

try:
    PeriodicSystem = __import__('PeriodicSystem').PeriodicSystem
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing PeriodicSystem: {str(e)}\n")
    del sys

try:
    BashScriptManager = __import__('BashScriptManager').BashScriptManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing BashScriptManager: {str(e)}\n")
    del sys

try:
    FileManager = __import__('FileManager').FileManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing FileManager: {str(e)}\n")
    del sys

try:
    import numpy as np
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing numpy: {str(e)}\n")
    del sys


class DFTSingleRun(FileManager): # el nombre no deberia incluir la palabra DFT tieneu qe ser ma general
    def __init__(self, file_location:str=None, name:str=None, **kwargs):
        super().__init__(name=name, file_location=file_location)

        self._containers = []  # Lista para almacenar subcontenedores
        self._KPointsManager = None
        self._AtomPositionManager = None
        self._Out_AtomPositionManager = None
        self._PotentialManager = None
        self._InputFileManager = None
        self._BashScriptManager = None

        self._loaded = {}

    def add_container(self, container):
        self.containers.append(container)

    def remove_container(self, container):
        self.containers.remove(container)

    def readVASPDirectory(self, file_location:str=None, v:bool=False):
        self.file_location = file_location if type(file_location) == str else self.file_location
        
        self.loadStoreManager(PeriodicSystem, 'POSCAR', '_AtomPositionManager', 'readPOSCAR', v)
        self.loadStoreManager(PotentialManager, 'POTCAR', '_PotentialManager', 'readPOTCAR', v)
        self.loadStoreManager(InputDFT, 'INCAR', '_InputFileManager', 'readINCAR', v)
        self.loadStoreManager(PeriodicSystem, 'CONTCAR', '_Out_AtomPositionManager', 'readCONTCAR', v)
        self.loadStoreManager(KPointsManager, 'KPOINTS', '_KPointsManager', 'readKPOINTS', v)
        self.loadStoreManager(BashScriptManager, 'VASPscript.sh', '_BashScriptManager', 'readBashScript', v)

    def exportVASP(self, file_location:str=None):
        file_location = file_location if type(file_location) == str else self.file_location
        self.create_directories_for_path(file_location)

        self.KPointsManager.exportAsKPOINTS(file_location+'/KPOINTS')
        self.AtomPositionManager.exportAsPOSCAR(file_location+'/POSCAR')
        self.PotentialManager.exportAsPOTCAR(file_location+'/POTCAR', self.AtomPositionManager.uniqueAtomLabels )
        self.InputFileManager.exportAsINCAR(file_location+'/INCAR')
        self.BashScriptManager.exportAsBash(file_location+'/VASPscript.sh')

    def recommend_vasp_parameters(self, num_cores, num_nodes, num_atoms:int=None, system_type:str="multi-core", network_type:str="fast"):
        """
        Recommends optimal VASP parameters based on the number of cores, nodes, and system type.

        Parameters:
            num_cores (int): Total number of compute cores available.
            num_nodes (int): Total number of nodes in the cluster.
            num_atoms (int): Number of atoms in the unit cell.
            system_type (str): Type of system ("multi-core" or "massive-parallel").
            network_type (str): Type of network ("fast" or "slow").

        Returns:
            dict: Recommended VASP parameters.
        """
        num_atoms = num_atoms if num_atoms is not None else self.AtomPositionManager.atomCount 
        recommendations = {}

        # General recommendations
        kpar = 1  # Default value
        if num_nodes > 1:
            kpar = num_nodes  # Usually set KPAR equal to the number of nodes for multi-node jobs

        # NCORE recommendations
        if system_type == "multi-core":
            ncore = min(4, num_cores // kpar)  # 2-4 is generally good for modern multi-core systems
        elif system_type == "massive-parallel":
            ncore = max(1, int((num_cores ** 0.5) // kpar))  # Square root of the total number of cores
        else:
            ncore = 1  # Default value for unknown system type

        # NPAR recommendations
        npar = max(1, num_cores // (kpar * ncore))

        # LPLANE and NSIM recommendations
        if network_type == "fast":
            lplane = ".TRUE."
            nsim = 4
        else:
            lplane = ".FALSE."
            nsim = 1

        # LSCALU recommendations
        lscalu = ".FALSE."

        # Populate recommendations dictionary
        recommendations["KPAR"] = kpar
        recommendations["NCORE"] = ncore
        recommendations["NPAR"] = npar
        recommendations["LPLANE"] = lplane
        recommendations["NSIM"] = nsim
        recommendations["LSCALU"] = lscalu

        return recommendations

    def estimate_vasp_memory(self, num_atoms:int=None, num_kpoints:int=None, num_bands:int=None, ncore:int=1):
        """
        Estimate the amount of RAM needed per node for a VASP calculation.

        Parameters:
            num_atoms (int): Number of atoms in the unit cell.
            num_bands (int): Number of electronic bands in the calculation.
            num_kpoints (int): Number of k-points in the Brillouin zone sampling.
            ncore (int): Number of cores working on each orbital (NCORE).

        Returns:
            float: Estimated RAM needed per node in GB.
        """

        num_bands = num_bands if num_bands is not None else self.estimateNumberOfBands()
        num_kpoints = num_kpoints if num_kpoints is not None else self.estimateNumberOfKPOINTS()
        num_atoms = num_atoms if num_atoms is not None else self.AtomPositionManager.atomCount

        # Estimate the RAM needed for wavefunctions (complex numbers, 8 bytes each)
        wavefunction_memory = num_atoms * num_bands * num_kpoints * 8  # in bytes

        # Convert to GB and multiply by NCORE
        memory_per_node_gb = (wavefunction_memory / (1024 ** 3)) * ncore  # in GB

        return memory_per_node_gb

    def estimate_vasp_runtime(self, num_cores, num_atoms:int=None, num_kpoints:int=None, num_bands:int=None, scaling_factor:float=1e-6):
        """
        Estimate the runtime needed for a VASP calculation.

        Parameters:
            num_atoms (int): Number of atoms in the unit cell.
            num_bands (int): Number of electronic bands in the calculation.
            num_kpoints (int): Number of k-points in the Brillouin zone sampling.
            num_cores (int): Number of cores used for the calculation.
            scaling_factor (float): An arbitrary scaling factor to fine-tune the estimate.

        Returns:
            float: Estimated runtime in hours.
        """

        # Estimate the runtime based on the number of atoms, bands, and k-points
        num_bands = num_bands if num_bands is not None else self.estimateNumberOfBands()
        num_kpoints = num_kpoints if num_kpoints is not None else self.estimateNumberOfKPOINTS()
        num_atoms = num_atoms if num_atoms is not None else self.AtomPositionManager.atomCount

        estimated_runtime = (num_atoms * num_bands * num_kpoints) / num_cores * scaling_factor  # in hours

        return estimated_runtime

    def estimateNumberOfBands(self, ):
        """
        Estimate the number of bands needed for a VASP calculation based on the POSCAR file content.

        Parameters:
            poscar_content (str): Content of the POSCAR file as a string.
            valence_electrons (dict): Dictionary mapping element symbols to the number of valence electrons.

        Returns:
            int: Estimated number of bands.
        """
        
        total_electrons = 0
        for element, count in zip(self.AtomPositionManager.uniqueAtomLabels , self.AtomPositionManager.atomCountByType):
            if element not in self.valenceElectrons:
                raise ValueError(f"Valence electrons for element {element} not provided.")
            total_electrons += self.valenceElectrons[element] * count

        num_bands = -(-total_electrons // 2)  # Equivalent to ceil(total_electrons / 2)
        
        return num_bands

    def estimateNumberOfKPOINTS(self,):
        return np.prod(self.KPointsManager.subdivisions)


'''
DSR = DFTSingleRun('/home/akaris/Documents/code/Physics/VASP/v6.1/files/bulk_optimization/Pt/FCC100')
DSR.readVASPDirectory(v=True)
print(DSR.recommend_vasp_parameters(num_cores=80, num_nodes=1))
print( DSR.estimateNumberOfBands() )

print(DSR.estimate_vasp_memory())
print(DSR.estimate_vasp_runtime(40))
print(DSR.containers)
'''