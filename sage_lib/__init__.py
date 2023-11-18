# __init__.py en el directorio sage_lib/

# Importaciones para facilitar el acceso a las clases y funciones más importantes
from .DFTPartition import DFTPartition
from .SurfaceStatesGenerator import SurfaceStatesGenerator
from .VacuumStatesGenerator import VacuumStatesGenerator

from .DFTSingleRun import DFTSingleRun
from .CrystalDefectGenerator import CrystalDefectGenerator
from .BashScriptManager import BashScriptManager
from .ChargeFileManager import ChargeFileManager
from .WaveFileManager import WaveFileManager
from .BinaryDataHandler import BinaryDataHandler

from .OutFileManager import OutFileManager

from .AtomPositionManager import AtomPositionManager
from .PeriodicSystem import PeriodicSystem
from .PotentialManager import PotentialManager
from .InputDFT import InputDFT
from .KPointsManager import KPointsManager

from .FileManager import FileManager

# Inicialización de variables, si es necesario
global_seed = 42

# Si desea controlar qué se importa con "from sage_lib import *"
__all__ = ["DFTPartition", "OutFileManager", "DFTSingleRun", "CrystalDefectGenerator"]

# Código de inicialización, si es necesario
def initialize_sage_lib():
    print("Inicializando sage_lib...")