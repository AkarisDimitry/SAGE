# En __init__.py del paquete que contiene AtomPositionManager
try:
    PeriodicSystem = __import__('PeriodicSystem').PeriodicSystem
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing PeriodicSystem: {str(e)}\n")
    del sys

try:
    import numpy as np
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing numpy: {str(e)}\n")
    del sys

class CrystalDefectGenerator(PeriodicSystem):
    def __init__(self, file_location:str=None, name:str=None, **kwargs):
        super().__init__(name=name, file_location=file_location)
        self._Vacancy = None 

    def introduce_vacancy(self, atom_index: int):
        """
        Introduce a vacancy by removing an atom.
        
        A vacancy is a type of point defect where an atom is missing from one of the lattice sites.
        The method modifies the atomic positions array and associated metadata to reflect the removal
        of an atom from the lattice.
        """
        self._atomPositions = np.delete(self._atomPositions, atom_index, axis=0)
        removed_atom_label = self._atomLabelsList.pop(atom_index)
        self._atomCount -= 1

        removed_atom_type_index = self._uniqueAtomLabels.index(removed_atom_label)
        self._atomCountByType[removed_atom_type_index] -= 1

    def introduce_vacancy(self, atom_index: int, tolerance=1e-5):
        """
        Introduce a vacancy by removing an atom.
        """
        # Remove the atom at the specified index
        atom_to_remove = self._atomPositions[atom_index]
        self._atomPositions = np.delete(self.atomPositions, atom_index, axis=0)
        removed_atom_label = self.atomLabelsList[atom_index]
        self._atomLabelsList = np.delete(self.atomLabelsList, atom_index)

        self._atomCount -= 1

        removed_atom_type_index = np.where(self.uniqueAtomLabels == removed_atom_label)[0][0]
        self.atomCountByType[removed_atom_type_index] -= 1
        
        # If the system is a surface, remove a symmetrically opposite atom
        if self.is_surface:
            # Calculate the center of the crystal, assuming it's at the origin for simplicity
            # If it's not, you would set center to the actual coordinates
            center = np.array([0, 0, 0])

            # Find the symmetrically opposite atom's position
            opposite_atom_position = 2 * center - atom_to_remove

            # Search for the index of the opposite atom using the tolerance criterion
            distances = np.linalg.norm(self._atomPositions - opposite_atom_position, axis=1)
            opposite_atom_index = np.where(distances < tolerance)[0]

            if len(opposite_atom_index) == 0:
                print("No symmetric atom found within tolerance.")
                return

            # Remove the opposite atom
            self._atomPositions = np.delete(self._atomPositions, opposite_atom_index, axis=0)
            removed_atom_label = self._atomLabelsList.pop(opposite_atom_index[0])
            self._atomCount -= 1

            removed_atom_type_index = self._uniqueAtomLabels.index(removed_atom_label)
            self._atomCountByType[removed_atom_type_index] -= 1


    def introduce_self_interstitial(self, position: np.array):
        """
        Introduce a self-interstitial defect.
        
        A self-interstitial is a type of point defect where an extra atom is added to an interstitial site.
        This method adds an atom to a specified interstitial position and updates the associated metadata.
        """
        self._atomPositions = np.vstack([self._atomPositions, position])
        self._atomLabelsList.append("SomeLabel")  # Replace with appropriate label
        self._atomCount += 1

        # Update _atomCountByType here similar to introduce_vacancy
        
    def introduce_substitutional_impurity(self, atom_index: int, new_atom_label: str):
        """
        Introduce a substitutional impurity.
        
        A substitutional impurity is a type of point defect where an atom is replaced by an atom of a different type.
        This method modifies the type of atom at the specified index to a new type.
        """
        self._atomLabelsList[atom_index] = new_atom_label

        # Update _atomCountByType here similar to introduce_vacancy

    def introduce_interstitial_impurity(self, position: np.array, atom_label: str):
        """
        Introduce an interstitial impurity.
        
        An interstitial impurity is a type of point defect where an atom of a different element is inserted at an interstitial site.
        This method adds an atom of a specified type to a specified interstitial position.
        """
        self._atomPositions = np.vstack([self._atomPositions, position])
        self._atomLabelsList.append(atom_label)
        self._atomCount += 1

        # Update _atomCountByType here similar to introduce_vacancy

    def generate_all_vacancies(self):
        """
        Generate all possible vacancy configurations for the system.
        
        A vacancy configuration is a unique arrangement of atoms that results from the removal of a single atom from the original structure.
        This method iterates through each atom in the system, removes it to create a vacancy, and then saves or outputs that new configuration.
        """
        
        # Initialize a list to hold all possible vacancy configurations
        all_vacancy_configs = []
        
        # Loop through each atom index to introduce a vacancy at that index
        for i in range(self._atomCount):
            
            # Clone the current object to preserve the original configuration
            temp_manager = copy.deepcopy(self)
            
            # Introduce a vacancy at index i
            temp_manager.introduce_vacancy(i)
            
            # Save or output the new configuration
            # For the purpose of this example, we'll append it to a list
            all_vacancy_configs.append(copy.deepcopy(temp_manager))
            
            # The original object remains unchanged, and can be used for the next iteration
        
        # Now, all_vacancy_configs contains all possible vacancy configurations.
        # You could choose to return them, write them to files, analyze them further, etc.
        return all_vacancy_configs


path = '/home/akaris/Documents/code/Physics/VASP/v6.1/files/POSCAR/Cristals/NiOOH/Bulk FeOOH with β-NiOOH structure (Fe(LS))'
path = '/home/akaris/Documents/code/Physics/VASP/v6.1/files/POSCAR/Cristals/NiOOH/Bulk FeOOH with β-NiOOH structure (Fe(HS))'
path = '/home/akaris/Documents/code/Physics/VASP/v6.1/files/POSCAR/Cristals/NiOOH/Bulk β-NiOOH doped with 1 Fe(HS)'

ap = CrystalDefectGenerator(file_location=path+'/SUPERCELL')
ap.readSIFile()
print( ap.latticeType )
ap.introduce_vacancy(atom_index=10)

ap.exportAsPOSCAR(path+'/POSCAR_d1')
