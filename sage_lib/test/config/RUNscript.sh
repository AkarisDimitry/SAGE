#!/bin/bash
#SBATCH --job-name=my_job_name        # Nombre del trabajo
#SBATCH --output=my_output_file.out   # Archivo de salida (stdout)
#SBATCH --error=my_error_file.err     # Archivo de errores (stderr)
#SBATCH --nodes=1                     # Número de nodos
#SBATCH --ntasks-per-node=72           # Número de tareas por nodo
#SBATCH --time=06:10:00               # Tiempo máximo (D-HH:MM)


# Ejecuta el script de Python
source activate SAGE

sage xyz --path . --all-subfolders --source VASP


