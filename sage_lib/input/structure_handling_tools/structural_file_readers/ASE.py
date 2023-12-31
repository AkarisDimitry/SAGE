try:
    from sage_lib.master.FileManager import FileManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing FileManager: {str(e)}\n")
    del sys

try:
    from sage_lib.master.AtomicProperties import AtomicProperties
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing AtomicProperties: {str(e)}\n")
    del sys

try:
    from ase import Atoms
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing numpy: {str(e)}\n")
    del sys

try:
    import numpy as np
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing numpy: {str(e)}\n")
    del sys

try:
    import re
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing re: {str(e)}\n")
    del sys

try:
    import pickle
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing pickle: {str(e)}\n")
    del sys

class ASE(Atoms, FileManager):
    """
    ASE class inherits from Atoms and FileManager, facilitating operations related
    to atomic structures and file management.
    """

    def __init__(self, name:str=None, file_location:str=None ):    
        """
        Initialize the ASE class by initializing parent classes.
        :param name: Name of the file.
        :param file_location: Location of the file.
        """
        FileManager.__init__(self, name=name, file_location=file_location)
        # Initialize Atoms with default values, including PBC
        Atoms.__init__(self, pbc=np.array([True, True, True]))
        self._pbc = np.array([True, True, True])  # Example PBC initialization

    def export_as_ASE(self, file_location:str=None, verbose:bool=False) -> bool:
        """
        Exports the ASE object to a specified file location.
        :param file_location: The file path to save the object.
        :param verbose: If True, enables verbose output.
        :return: Boolean indicating successful export.
        """
        file_location = file_location if file_location is not None else 'ASE.obj'

        try:
            with open(file_location, 'wb') as file:
                pickle.dump(self, file)
            return True
        except Exception as e:
            if verbose:
                print(f"Error exporting ASE object: {e}")
            return False

    def read_ASE(self, ase_atoms:object=None, file_location:str=None):
        """
        Reads an ASE object either from an existing Atoms object or from a file.
        :param ase_atoms: An existing Atoms object.
        :param file_location: The file path to read the object from.
        :return: Boolean indicating successful read.
        """
        if ase_atoms is not None:
            self.ASE_2_SAGE(ase_atoms=ase_atoms)
            return True

        elif file_location is not None: 
            from ase.io import read
            self = read(file_path)
            self.ASE_2_SAGE(ase_atoms=ase_atoms)
            return True

        return False

    def ASE_2_SAGE(self, ase_atoms:Atoms=None):
        """
        Transforms an ASE Atoms object to the SAGE internal representation.
        :param ase_atoms: An ASE Atoms object.
        """

        # Configuración básica
        self._atomCount = len(ase_atoms)
        self._uniqueAtomLabels = list(set(ase_atoms.get_chemical_symbols()))
        self._atomCountByType = [ase_atoms.get_chemical_symbols().count(x) for x in self._uniqueAtomLabels]
        self._atomPositions = ase_atoms.get_positions()
        self._atomLabelsList = ase_atoms.get_chemical_symbols()
        self._latticeVectors = ase_atoms.get_cell()
        self._cellVolumen = ase_atoms.get_volume()
        self._pbc = ase_atoms.get_pbc()
        self._atomCoordinateType = 'Cartesian'
        # Energías y Fuerzas
        # Nota: Estas propiedades dependen de si han sido calculadas y almacenadas en el objeto Atoms

        try:
            self._E = ase_atoms.get_total_energy() 
        except (AttributeError, RuntimeError):
            self._E = None  # O algún valor predeterminado si es más apropiado

        try:
            self._K = ase_atoms.get_kinetic_energy()
        except (AttributeError, RuntimeError):
            self._K = None

        # Para la energía potencial, verifica si tanto E como K están disponibles
        if self._E is not None and self._K is not None:
            self._U = self._E - self._K
        else:
            self._U = None  # O algún valor predeterminado
            
        self._total_force = np.sum(ase_atoms.get_forces(), axis=0) if 'forces' in ase_atoms.arrays else None

