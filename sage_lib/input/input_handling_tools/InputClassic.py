try:
    from sage_lib.input.input_handling_tools.InputFileManager import InputFileManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing sage_lib.input.input_handling_tools.InputFileManager: {str(e)}\n")
    del sys

try:
    import numpy as np
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing numpy: {str(e)}\n")
    del sys

class InputClassic(InputFileManager):
    def __init__(self, file_location:str=None, name:str=None, **kwargs):
        super().__init__(name=name, file_location=file_location)
        pass
