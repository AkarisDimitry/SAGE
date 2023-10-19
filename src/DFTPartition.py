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
        directories = []

        for c in self.containers:
            for v in values:
                cCopy = copy.deepcopy(c)

                if parameter.upper() == 'KPOINTS':
                    cCopy.file_location = c.file_location + f'/KPOINTConvergence/{v[0]}_{v[1]}_{v[2]}' if file_location == None else file_location
                    cCopy.KPointsManager.subdivisions = v

                if parameter.upper() == 'ENCUT':
                    cCopy.file_location = c.file_location + f'/ENCUTConvergence/{v}' if file_location == None else file_location
                    cCopy.InputFileManager.parameters['ENCUT'] = v

                cCopy.KPointsManager.file_location = v
                containers.append( cCopy )
                directories.append( cCopy.file_location )

        self.containers = containers
        self.generate_python_script_for_sbatch(directories, self.file_location)
        
    def writePartition(self, file_location:str=None): 
        for c in self.containers:
            c.exportVASP()

    # Function to generate a Python script for running sbatch in multiple directories
    def generate_python_script_for_sbatch(self, directories:list=None, file_location:str=None):
        script_content = '''#!/usr/bin/env python3
import os
import subprocess

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
    os.chdir('..')
                            '''

        # Write the script to a file
        with open(f"{file_location}/run_sbatch_in_directories.py", "w") as f:
            f.write(script_content)

#DP = DFTPartition('/home/akaris/Documents/code/Physics/VASP/v6.1/files/bulk_optimization/Pt/parametros/ENCUT_optimization_252525_FCC')
DP = DFTPartition('/home/akaris/Documents/code/Physics/VASP/v6.1/files/bulk_optimization/Pt/FCC100')
#DP.readVASPSubFolder(v=True)
DP.readVASPFolder(v=True)
DP.generateDFTVariants('KPOINTS', [[n,n,n] for n in range(20)] )
DP.writePartition()
'''
DP.generateDFTVariants('ENCUT', [ E for E in range(400,700,30)] )
print( DP.summary() )

print( DP.containers[0].AtomPositionManager.summary() )
'''

