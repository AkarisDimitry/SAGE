try:
    from sage_lib.input.structure_handling_tools.AtomPosition import AtomPosition
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing AtomPosition: {str(e)}\n")
    del sys
    
try:
    from sage_lib.input.input_handling_tools.InputFile import InputFile
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing InputFile: {str(e)}\n")
    del sys

try:
    from sage_lib.input.BinaryDataHandler import BinaryDataHandler
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing BinaryDataHandler: {str(e)}\n")
    del sys

try: 
    from sage_lib.input.PotentialManager import PotentialManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing PotentialManager: {str(e)}\n")
    del sys

try:
    from sage_lib.input.KPointsManager import KPointsManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing KPointsManager: {str(e)}\n")
    del sys

try:
    from sage_lib.input.BashScriptManager import BashScriptManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing BashScriptManager: {str(e)}\n")
    del sys

try:
    from sage_lib.output.OutFileManager import OutFileManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing OutFileManager: {str(e)}\n")
    del sys

try:
    from sage_lib.output.WaveFileManager import WaveFileManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing WaveFileManager: {str(e)}\n")
    del sys

try:
    from sage_lib.output.ChargeFileManager import ChargeFileManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing ChargeFileManager: {str(e)}\n")
    del sys

try:
    from sage_lib.master.FileManager import FileManager
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

class SingleRunManager(FileManager): # el nombre no deberia incluir la palabra DFT tieneu qe ser ma general
    def __init__(self, file_location:str=None, name:str=None, **kwargs):
        super().__init__(name=name, file_location=file_location)
        self._AtomPositionManager_constructor = AtomPosition
        self._Out_AtomPositionManager_constructor = AtomPosition
        self._PotentialManager_constructor = PotentialManager
        self._InputFileManager_constructor = InputFile
        self._KPointsManager_constructor = KPointsManager
        self._BashScriptManager_constructor = BashScriptManager
        self._BinaryDataHandler_constructor = BinaryDataHandler

        self._OutFileManager_constructor = OutFileManager
        self._WaveFileManager_constructor = WaveFileManager
        self._ChargeFileManager_constructor = ChargeFileManager

    def NonPeriodic_2_Periodic(self, ):
        if self.AtomPositionManager is NonPeriodicSystem:
            PS = PeriodicSystem(**self.AtomPositionManager)
            self.atomPositions -= np.mean(self.atomPositions, axis=1) 

        return True
