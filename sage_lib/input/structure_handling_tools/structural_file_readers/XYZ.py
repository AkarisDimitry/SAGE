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

try:
    import re
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing re: {str(e)}\n")
    del sys

class XYZ(FileManager):
    def __init__(self, file_location:str=None, name:str=None, **kwargs):
        super().__init__(name=name, file_location=file_location)

    def export_as_xyz(self, file_location:str=None, save_to_file:str='w', verbose:bool=False) -> str:
        """
        Export atomistic information in the XYZ format.

        Parameters:
            file_location (str): The location where the XYZ file will be saved. Ignored if save_to_file is False.
            save_to_file (bool): Flag to control whether to save the XYZ content to a file.
            verbose (bool): Flag to print additional information, if True.

        Returns:
            str: The generated XYZ content.
        """
        file_location  = file_location  if not file_location  is None else self.file_location+'config.xyz' if self.file_location is str else self.file_location
        self.group_elements_and_positions()

        # Initialize an empty string to store the XYZ content
        xyz_content = ""

        # Write the number of atoms
        xyz_content += f"{self.atomCount}\n"

        # Write information about the unit cell, energy, etc.
        lattice_str = " ".join(map(str, self._latticeVectors.flatten()))
        #pbc_str = ' '.join(['T' if val else 'F' for val in self.pbc])
        xyz_content += f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:DFT_forces:R:3 DFT_energy={self._E}  pbc="T T T"\n'

        # Column widths for alignment. These can be adjusted.
        col_widths = {'element': 5, 'position': 12, 'force': 12}
        # Write the atom positions, masses, and forces
        for i in range(self.atomCount):
            atom_label = self.atomLabelsList[i]
            pos = " ".join(map("{:12.6f}".format, self.atomPositions[i])) if self.atomPositions is not None else ''
            force = " ".join(map("{:14.6f}".format, self.total_force[i])) if self.total_force is not None else ''  # Assuming that self._total_force is an array (N, 3)
            xyz_content += f"{atom_label:<{col_widths['element']}}{pos}{force}\n"

        # Save the generated XYZ content to a file if file_location is specified and save_to_file is True
        if file_location and save_to_file:
            with open(file_location, save_to_file) as f:
                f.write(xyz_content)
            if verbose:
                print(f"XYZ content has been saved to {file_location}")

        return xyz_content

    def read_XYZ(self, file_location: str = None, lines: list = None, energy_key: str = 'energy', 
                       masses_key: str = 'masses', forces_key: str = 'forces', position_key: str = 'pos',  
                       species_key: str = 'species', PBC_key: str = 'pbc', verbose: bool = False):
        """
        Reads and parses data from a XYZ configuration file used in molecular simulations.

        :param file_location: Location of the XYZ file. If None, uses default file location.
        :param lines: List of lines from the file to be read. If None, reads from the file directly.
        :param energy_key: Key string for energy data in the file.
        :param masses_key: Key string for atomic masses data in the file.
        :param forces_key: Key string for forces data in the file.
        :param position_key: Key string for atomic positions data in the file.
        :param species_key: Key string for atomic species data in the file.
        :param PBC_key: Key string for periodic boundary conditions data in the file.
        :param verbose: If True, prints additional information during processing.
        :return: True if the file is successfully read and parsed, False otherwise.
        """        
        file_location = file_location if isinstance(file_location, str) else self.file_location

        pattern = r'(\w+)=("[^"]+"|\S+)'
        data = {'species', 'pos', 'masses', 'forces', 'E'}
        read_header = False


        lines = lines if lines is not None else list(self.read_file(file_location,strip=False))

        for i, line in enumerate(lines):
            if read_header:
                matches = re.findall(pattern, line)
                body = np.array( [ n.strip().split() for n in lines[i+1:i+self._atomCount+1] ])
                for key, value in matches:
                    if key == 'Lattice':
                        self._latticeVectors = np.array([ [ float(value[1:-1].strip().split()[i*3+j]) for j in range(3) ] for i in range(3) ])
                        self._atomCoordinateType = 'C'
                        
                    if key == 'Properties':
                        matches_Properties_count = 0
                        matches_Properties_vec = value.split(':')
                        for pi, p in enumerate(matches_Properties_vec):
                            if forces_key in p:
                                self._total_force = np.array(body[:, matches_Properties_count:matches_Properties_count+3], dtype=np.float64)

                            elif masses_key in p:
                                self._mass = body[:, matches_Properties_count]

                            elif position_key in p:
                                self._atomPositions = np.array(body[:, matches_Properties_count:matches_Properties_count+3], dtype=np.float64)

                            elif species_key in p:
                                self._atomLabelsList = body[:, matches_Properties_count]

                            if pi%3==0:
                                matches_Properties_count += int(matches_Properties_vec[pi+2])

                    if key == energy_key:
                        self._E = float(value)

                    if key == PBC_key:
                        self._pbc = list( ['T' in v  for v in value.split()] )

                return True

            elif line.strip().isdigit():
                num_atoms = int(line.strip())
                if num_atoms > 0:
                    self._atomCount = num_atoms
                    read_header = True 