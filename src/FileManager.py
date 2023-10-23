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
    from datetime import datetime
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing datetime: {str(e)}\n")
    del sys

class FileManager:
    def __init__(self, file_location:str=None, name:str=None):
        if name is not None and not isinstance(name, str):
            raise TypeError("(!) name must be a string")
        self._name = name
        self._metaData = {}
        self._comment = ''

        if file_location is not None and not isinstance(file_location, str):
            raise TypeError("(!) file_location must be a string")
        self._file_location = file_location   

        self.plot_color = [ # pastel
            '#FFABAB',  # Salmon (Pastel)       #FFABAB    (255,171,171)
            '#A0C4FF',  # Sky Blue (Pastel)     #A0C4FF    (160,196,255)
            '#B4F8C8',  # Mint (Pastel)         #B4F8C8    (180,248,200)
            '#FFE156',  # Yellow (Pastel)       #FFE156    (255,225,86)
            '#FBE7C6',  # Peach (Pastel)        #FBE7C6    (251,231,198)
            '#AB83A1',  # Mauve (Pastel)        #AB83A1    (171,131,161)
            '#6C5B7B',  # Thistle (Pastel)      #6C5B7B    (108,91,123)
            '#FFD1DC',  # Pink (Pastel)         #FFD1DC    (255,209,220)
            '#392F5A',  # Purple (Pastel)       #392F5A    (57,47,90)
            '#FF677D',  # Watermelon (Pastel)   #FF677D    (255,103,125)
            '#FFC3A0',  # Coral (Pastel)        #FFC3A0    (255,195,160)
            '#6A057F',  # Lavender (Pastel)     #6A057F    (106,5,127)
            '#D4A5A5',  # Rose (Pastel)         #D4A5A5    (212,165,165)
            '#ACD8AA',  # Sage (Pastel)         #ACD8AA    (172,216,170)
        ]

        self.valenceElectrons = {
                "H": 1, "He": 2,
                "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "Ne": 8,
                "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7, "Ar": 8,
                "K": 1, "Ca": 2, "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11, "Zn": 12,
                "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7, "Kr": 8,
                "Rb": 1, "Sr": 2, "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Tc": 7, "Ru": 8, "Rh": 9, "Pd": 10, "Ag": 11, "Cd": 12,
                "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7, "Xe": 8,
                "Cs": 1, "Ba": 2, "La": 3, "Ce": 4, "Pr": 5, "Nd": 6, "Pm": 7, "Sm": 8, "Eu": 9, "Gd": 10, "Tb": 11, "Dy": 12, 
                "Ho": 13, "Er": 14, "Tm": 15, "Yb": 16, "Lu": 17, "Hf": 4, "Ta": 5, "W": 6, "Re": 7, "Os": 8, "Ir": 9, 
                "Pt": 10, "Au": 11, "Hg": 12, "Tl": 13, "Pb": 14, "Bi": 15, "Th": 16, "Pa": 17, "U": 18, "Np": 19, "Pu": 20
                                }
        self._FM_attrs = set(vars(self).keys())

    def __getattr__(self, name):
        # Check if the attribute exists with a leading underscore
        private_name = f"_{name}"
        if private_name in self.__dict__:
            return getattr(self, private_name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Check if the attribute exists with a leading underscore
        private_name = f"_{name}"
        if private_name in self.__dict__:
            setattr(self, private_name, value)
        else:
            super().__setattr__(name, value)
            
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError("(!) name must be a string")
        self._name = value

    @name.deleter
    def name(self):
        print("Deleting name")
        del self._name

    @property
    def file_location(self):
        return self._file_location

    @file_location.setter
    def file_location(self, value):
        if not isinstance(value, str):
            raise TypeError("(!) file_location must be a string")
        self._file_location = value

    @file_location.deleter
    def file_location(self):
        print("Deleting file_location")
        del self._file_location

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def isINT(self, num): return self.is_number(num) and abs(num - int(num)) < 0.0001 

    def file_exists(self, file_path:str) -> bool:
        """Check if a file exists."""
        return os.path.exists(file_path)

    def create_directories_for_path(self, path:str):
        """
        Create any directories needed to ensure that the given path is valid.

        Parameters:
        - path (str): The file or directory path to validate.
        """
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
            except PermissionError:
                print(f"Permission denied: Could not create directory at {path}")
                # Exit or handle the error appropriately
                exit(1)

    def read_file(self, file_location:str=None, strip=True):
        file_location = file_location if type(file_location) == str else self._file_location

        if file_location is None:
            raise ValueError("(!) File location is not set.")
        
        try:
            with open(file_location, 'r') as file:
                if strip:
                    for line in file:
                        yield line.strip()  # strip() elimina espacios y saltos de línea al principio y al final
                else:
                    for line in file:
                        yield line  # strip() elimina espacios y saltos de línea al principio y al final
        except FileNotFoundError:
            raise FileNotFoundError(f"(!) File not found at {file_location}")
        except IOError:
            raise IOError(f"(!) Error reading file at {file_location}")
        except Exception as e:
            print(f"Error inesperado: {e}")

    def loadStoreManager(self, manager_class, file_name:str, attribute_name: str, read_method:str, v:bool=False):
        """Generic function to read a VASP file."""
        file_path = f"{self.file_location}/{file_name}"
        if not self.file_exists(file_path):
            return
        
        try:
            manager = manager_class(file_path)
            getattr(manager, read_method)()  # Call the appropriate read method
            setattr(self, attribute_name, manager)
            self._loaded[file_name] = True
            if v:
                print(manager.summary())
        except Exception as e:
            if v:
                print(f"ERROR :: Cannot load {file_name}. Reason: {e}")
