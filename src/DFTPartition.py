try:
    DFTSingleRun = __import__('DFTSingleRun').DFTSingleRun
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing DFTSingleRun: {str(e)}\n")
    del sys

try:
    FileManager = __import__('FileManager').FileManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing FileManager: {str(e)}\n")
    del sys

try:
    CrystalDefectGenerator = __import__('CrystalDefectGenerator').CrystalDefectGenerator
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing CrystalDefectGenerator: {str(e)}\n")
    del sys

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
    import copy
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing copy: {str(e)}\n")
    del sys

class DFTPartition(FileManager): # el nombre no deberia incluir la palabra DFT tieneu qe ser ma general
    def __init__(self, file_location:str=None, name:str=None, **kwargs):
        super().__init__(name=name, file_location=file_location)
        self._containers = []

    def add_container(self, container:object):
        self.containers.append(container)

    def remove_container(self, container:object):
        self.containers.remove(container)

    def empty_container(self, container:object):
        self.containers = []

    def readVASPSubFolder(self, file_location:str=None, v=False):
        file_location = file_location if type(file_location) == str else self.file_location
                
        for root, dirs, files in os.walk(file_location):
            DFT_SR = self.readVASPFolder(root, v)
            if v: print(root, dirs, files)

    def readVASPFolder(self, file_location:str=None, v=False):
        file_location = file_location if type(file_location) == str else self.file_location

        DFT_SR = DFTSingleRun(file_location)
        DFT_SR.readVASPDirectory()        
        self.add_container(container=DFT_SR)
        return DFT_SR

    def summary(self, ) -> str:
        text_str = ''
        text_str += f'{self.file_location}\n'
        text_str += f'> Conteiners : { len(self.containers) }\n'
        return text_str

    def generateDFTVariants(self, parameter:str, values:np.array, file_location:str=None) -> bool:
        containers = []
        directories = ['' for n in self.containers]

        for DFTSR_i, DFTSR in enumerate(self.containers): # DFTSingleRun
            sub_directories = []

            for v in values:
                if parameter.upper() == 'KPOINTS':
                    python_script_directory = DFTSR.file_location + '/KPOINTConvergence'
                    DFTSR_copy = copy.deepcopy(DFTSR)
                    DFTSR_copy.file_location  = DFTSR.file_location + f'/KPOINTConvergence/{v[0]}_{v[1]}_{1}' if file_location == None else file_location
                    DFTSR_copy.KPointsManager.subdivisions = [v[0], v[1], 1]
                    sub_directories.append( f'{v[0]}_{v[1]}_{1}' )
                    directories[DFTSR_i] = f'KPOINTConvergence' 
                    containers.append( DFTSR_copy )

                if parameter.upper() == 'ENCUT':
                    DFTSR_copy = copy.deepcopy(DFTSR)
                    DFTSR_copy.file_location = DFTSR.file_location + f'/ENCUTConvergence/{v}' if file_location == None else file_location
                    DFTSR_copy.InputFileManager.parameters['ENCUT'] = v
                    sub_directories.append( f'{v}' )
                    directories[DFTSR_i] = f'ENCUTConvergence' 
                    python_script_directory = DFTSR.file_location + '/ENCUTConvergence'
                    containers.append( DFTSR_copy )

                if parameter.upper() == 'VACANCY':
                    DFTSR_copy = copy.deepcopy(DFTSR)
                    DFTSR_copy.file_location = DFTSR.file_location + f'/Vacancy' if file_location == None else file_location
                    python_script_directory = DFTSR.file_location + '/Vacancy'

                    DFTSR_copy.AtomPositionManager = CrystalDefectGenerator(Periodic_Object=DFTSR_copy.AtomPositionManager)
                    DFTSR_copy.AtomPositionManager._is_surface = True
                    all_vacancy_configs, all_vacancy_label = DFTSR_copy.AtomPositionManager.generate_all_vacancies()

                    for cv_i, (vacancy_configs, vacancy_label) in enumerate(zip(all_vacancy_configs, all_vacancy_label)):
                        DFTSR_copy2 = copy.deepcopy(DFTSR_copy)
                        DFTSR_copy2.AtomPositionManager = vacancy_configs
                        DFTSR_copy2.file_location = DFTSR_copy.file_location + f'/{cv_i}_{vacancy_label}' if file_location == None else file_location

                        sub_directories.append( f'{cv_i}_{vacancy_label}' )

                        containers.append( DFTSR_copy2 )

                    directories[DFTSR_i] = f'Vacancy' 

            self.generate_python_script_for_sbatch(sub_directories, python_script_directory)
        
        self.generate_python_script_for_python(directories, DFTSR.file_location)
        
        self.containers = containers
        
    def writePartition(self, file_location:str=None): 
        for DFTSR in self.containers:
            print(DFTSR.estimate_vasp_memory() )
            print(DFTSR.recommend_vasp_parameters(40, 1) )
            DFTSR.exportVASP()

    # Function to generate a Python script for running sbatch in multiple directories
    def generate_python_script_for_sbatch(self, directories:list=None, file_location:str=None):
        self.create_directories_for_path(file_location)
        script_content = '''#!/usr/bin/env python3
import os
import subprocess

original_directory = os.getcwd()

directories = [ \n'''

        # Add directory list
        for directory in directories:
            script_content += f"    '{directory}',\n"

        script_content += '''\
                            ]

for directory in directories:
    os.chdir(directory)
    subprocess.run(['chmod', '+x', 'VASPscript.sh'])
    subprocess.run(['sbatch', 'VASPscript.sh'])
    os.chdir(original_directory)
                            '''

        # Write the script to a file
        with open(f"{file_location}/run_sbatch_in_directories.py", "w") as f:
            f.write(script_content)

    # Function to generate a Python script for running sbatch in multiple directories
    def generate_python_script_for_python(self, directories:list=None, file_location:str=None):
        self.create_directories_for_path(file_location)
        script_content = '''#!/usr/bin/env python3
import os
import subprocess

original_directory = os.getcwd()

directories = [ \n'''

        # Add directory list
        for directory in directories:
            script_content += f"    '{directory}',\n"

        script_content += '''\
                            ]

for directory in directories:
    os.chdir(directory)
    subprocess.run(['sbatch', 'run_sbatch_in_directories.py'])
    os.chdir(original_directory)
                            '''

        # Write the script to a file
        with open(f"{file_location}/run_python_in_directories.py", "w") as f:
            f.write(script_content)

path = '/home/akaris/Documents/code/Physics/VASP/v6.1/files/POSCAR/Cristals/NiOOH/*OH surface with Fe(HS)'
path = '/home/akaris/Documents/code/Physics/VASP/v6.1/files/POSCAR/Cristals/NiOOH/*OH surface for pure NiOOH'
#DP = DFTPartition('/home/akaris/Documents/code/Physics/VASP/v6.1/files/bulk_optimization/Pt/parametros/ENCUT_optimization_252525_FCC')
DP = DFTPartition(path)

#DP.readVASPSubFolder(v=True)
DP.readVASPFolder(v=True)
DP.generateDFTVariants('Vacancy', [1])
#DP.generateDFTVariants('KPOINTS', [[n,n,n] for n in range(1, 15)] )
#DP.generateDFTVariants('ENCUT', [n for n in range(200, 1100, 45)] )

DP.writePartition()
'''
DP.generateDFTVariants('ENCUT', [ E for E in range(400,700,30)] )
print( DP.summary() )

print( DP.containers[0].AtomPositionManager.summary() )
'''

