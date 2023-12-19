#!/usr/bin/env python3
import os
import subprocess

original_directory = os.getcwd()

directories = [
    'compress',
]

for directory in directories:
    os.chdir(directory)
    subprocess.run(['chmod', '+x', 'execution_script_for_each_container.py'])
    subprocess.run(['sbatch', 'execution_script_for_each_container.py'])
    os.chdir(original_directory)
