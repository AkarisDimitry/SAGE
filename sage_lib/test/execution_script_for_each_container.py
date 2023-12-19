#!/usr/bin/env python3
import os
import subprocess

original_directory = os.getcwd()

directories = [
    'OUT/export.xyz/000',
    'OUT/export.xyz/001',
    'OUT/export.xyz/002',
    'OUT/export.xyz/003',
    'OUT/export.xyz/004',
    'OUT/export.xyz/005',
    'OUT/export.xyz/006',
    'OUT/export.xyz/007',
    'OUT/export.xyz/008',
    'OUT/export.xyz/009',
    'OUT/export.xyz/010',
    'OUT/export.xyz/011',
    'OUT/export.xyz/012',
    'OUT/export.xyz/013',
    'OUT/export.xyz/014',
    'OUT/export.xyz/015',
]

for directory in directories:
    os.chdir(directory)
    subprocess.run(['chmod', '+x', 'RUNscript.sh'])
    subprocess.run(['sbatch', 'RUNscript.sh'])
    os.chdir(original_directory)
