try:
    from sage_lib.single_run.SingleRunDFT import SingleRunDFT
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing SingleRunDFT: {str(e)}\n")
    del sys
    
try:
    import numpy as np
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing numpy: {str(e)}\n")
    del sys

class SingleRun(SingleRunDFT): # el nombre no deberia incluir la palabra DFT tieneu qe ser ma general
    def __init__(self, file_location:str=None, name:str=None, **kwargs):
        super().__init__(name=name, file_location=file_location)



