#!/usr/bin/env python3
import os
import subprocess

original_directory = os.getcwd()

directories = [
    '/compress/N_compress_min_compress_max',
    '/compress/N_compress_min_compress_max',
    '/compress/N_compress_min_compress_max',
    '/compress/N_compress_min_compress_max',
    '/compress/N_compress_min_compress_max',
    '/compress/N_compress_min_compress_max',
    '/compress/N_compress_min_compress_max',
    '/compress/N_compress_min_compress_max',
    '/compress/N_compress_min_compress_max',
    '/compress/N_compress_min_compress_max',
]

for directory in directories:
    os.chdir(directory)
    subprocess.run(['chmod', '+x', 'RUNscript.sh'])
    subprocess.run(['sbatch', 'RUNscript.sh'])
    os.chdir(original_directory)
