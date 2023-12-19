#!/usr/bin/env python3
import os
import subprocess

original_directory = os.getcwd()

directories = [
    '/0',
    '/1',
    '/2',
    '/3',
    '/4',
    '/5',
    '/6',
    '/7',
    '/8',
    '/9',
    '/10',
    '/11',
    '/12',
    '/13',
    '/14',
    '/15',
    '/16',
    '/17',
    '/18',
    '/19',
]

for directory in directories:
    os.chdir(directory)
    subprocess.run(['chmod', '+x', 'RUNscript.sh'])
    subprocess.run(['sbatch', 'RUNscript.sh'])
    os.chdir(original_directory)
