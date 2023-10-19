# *** warning supresion
import warnings, os
warnings.filterwarnings("ignore")

# *** warning supresion
import warnings; warnings.filterwarnings("ignore")
import argparse

# *** numpy libraries
import numpy as np 	
import matplotlib.pyplot as plt
try:	
	from ase.visualize import view
	from ase import Atoms
	from ase.visualize.plot import plot_atoms
except: print('can not load ASE module. (pipX install ase-atomistics)')

try:	from src import Logs
except:	
	try: import Logs as Logs
	except: print('WARNING :: Set.import_libraries() :: can not import ORR ')

try:	from src import POSCAR
except:	
	try: import POSCAR as POSCAR
	except: print('WARNING :: POSCAR.import_libraries() :: can not import POSCAR ')

try:	from src import INCAR
except:	
	try: import INCAR as INCAR
	except: print('WARNING :: INCAR.import_libraries() :: can not import POSCAR ')

try:	from src import POTCAR
except:	
	try: import POTCAR as POTCAR
	except: print('WARNING :: POTCAR.import_libraries() :: can not import POSCAR ')

try:	from src import KPOINTS
except:	
	try: import KPOINTS as KPOINTS
	except: print('WARNING :: KPOINTS.import_libraries() :: can not import POSCAR ')

try:	from src import OSZICAR
except:	
	try: import OSZICAR as OSZICAR
	except: print('WARNING :: OSZICAR.import_libraries() :: can not import POSCAR ')

class SLAB(object):
	def __init__(self, name=None, atoms=None, atoms_list=None):
		self.name = None

		self.POSCAR  = None
		self.KPOINTS = None
		self.INCAR   = None
		self.POTCAR  = None

	def create(self, L:float=None, atoms_dict:dict=None, cell:list=None, save:bool=True) -> bool:
		# === gen new POSCAR ===  #
		poscar = POSCAR.POSCAR('SLAB')

		N = 0 						# (A) N total number of atoms :: len(self.atoms)
		scale = L 					# (B) scale factor
		selective_dynamics = True 	# (C) selective dynamics" feature  :: (bool)
		
		atoms = [] 					# (D) np.array(3,N)
		atoms_number = [] 			# (E) [n(Fe), n(N), n(C), n(H) ]

		contrains = []				# (F) np.array(3,N)  allow atom move during ion relaxation

		atoms_names_list = [] 	    # (G) [Fe, N, N, N, N, C, C, C, C, H]
		atoms_names_ID = [] 		# (H) [Fe, N, C, H] 
		atoms_names_full = ''		# (I) FeFeFeNNNNNNNCCCCCCCCCCCCCCCHHHHHHHHHHHHHHHH
		
		cell = cell  				# (J) 3x3 [a1,a2,a3]
		coordenate = 'Direct'  		# (K) Direct / Cartesian

		for atom_id, atoms_position in atoms_dict.items():
			atoms_names_ID.append(atom_id)
			atoms_number.append( len(atoms_position) )

			for atoms_position_n in atoms_position:
				N += 1
				atoms.append( atoms_position_n )
				contrains.append([True, True, True])
				atoms_names_list.append(atom_id)
				atoms_names_full += str(atom_id) 

		atoms = np.array(atoms)
		cell  = np.array(cell)

		if save:
			poscar.N = N 									# (A) N total number of atoms :: len(self.atoms)
			poscar.scale = scale 							# (B) scale factor
			poscar.selective_dynamics = selective_dynamics 	# (C) selective dynamics" feature  :: (bool)

			poscar.atoms = atoms 							# (D) np.array(3,N)
			poscar.atoms_number = atoms_number 				# (E) [n(Fe), n(N), n(C), n(H) ]

			poscar.contrains = contrains  					# (F) np.array(3,N)  allow atom move during ion relaxation

			poscar.atoms_names_list = atoms_names_list 		# (G) [Fe, N, N, N, N, C, C, C, C, H]
			poscar.atoms_names_ID = atoms_names_ID 			# (H) [Fe, N, C, H] 
			poscar.atoms_names_full = atoms_names_full 		# (I) FeFeFeNNNNNNNCCCCCCCCCCCCCCCHHHHHHHHHHHHHHHH
		
			poscar.cell = cell 								# (J) 3x3 [a1,a2,a3]
			poscar.coordenate = coordenate 					# (K) Direct / Cartesian

			self.POSCAR = poscar

		return True

	def load_files(self, path:str, files_list:list=['POTCAR', 'INCAR', 'KPOINTS', 'POSCAR']):
		files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
		print(files)

		for f in files_list:
			if 'POSCAR' in f and 'POSCAR' in files:
				self.POSCAR = POSCAR.POSCAR(f'{path}/POSCAR')
				print(self.POSCAR.file_name)
				self.POSCAR.load()
				#self.POTCAR.summary()

			if 'POTCAR' in f and 'POTCAR' in files:
				self.POTCAR = POTCAR.POTCAR(f'{path}/POTCAR')
				print(self.POTCAR.file_name)
				self.POTCAR.load()
				#self.POTCAR.summary()

			if 'INCAR' in f and 'INCAR' in files:
				self.INCAR = INCAR.INCAR(f'{path}/INCAR')
				self.INCAR.load()
				#self.INCAR.view()
				
			if 'KPOINTS' in f and 'KPOINTS' in files:
				self.KPOINTS = KPOINTS.KPOINTS(f'{path}/KPOINTS')
				self.KPOINTS.load()
				#self.KPOINTS.view()

	def variable_analysis(self, variable:str=None, values:list=None, path:str=None):
		print(f'{path}/lattice_optimization')
		# make analysis dir 
		os.makedirs(f'{path}/{variable}_optimization', exist_ok=True)
		for i, v in enumerate(values):
			# change scale in each iteration  
			if variable == 'lattice': self.POSCAR.scale = v
			if variable == 'ENCUT':   self.INCAR.var_assing('ENCUT', v)
			if variable == 'KPOINTS':   self.KPOINTS.subdivisions = v
			print(v)

			print(self.INCAR.ENCUT)
			# make particular dir 
			if type(v) in [float, int]:		os.makedirs(f'{path}/{variable}_optimization/{i}-{v:.3}', exist_ok=True)
			elif variable == 'KPOINTS':		
				v = float(v[0])
				os.makedirs(f'{path}/{variable}_optimization/{i}-{v}', exist_ok=True)

			# save POSCAR 
			self.POSCAR.export(f'{path}/{variable}_optimization/{i}-{v:.3}/POSCAR')
			# save KPOINTS 
			if not type(self.KPOINTS) is None: self.KPOINTS.save(f'{path}/{variable}_optimization/{i}-{v:.3}/KPOINTS')
			# save POTCAR 
			if not type(self.POTCAR) is None: self.POTCAR.save(f'{path}/{variable}_optimization/{i}-{v:.3}/POTCAR')
			# save INCAR 
			if not type(self.INCAR) is None: self.INCAR.save(f'{path}/{variable}_optimization/{i}-{v:.3}/INCAR')


	def analysis(self, variables:list=None, path:str=None, extract:str='energy') -> list:

		data = []
		if variables is None:
			for i, d in	enumerate( next(os.walk(path))[1]):
				if extract == 'energy':
					self.OSZICAR = OSZICAR.OSZICAR(f'{path}/{d}/OSZICAR')
					self.OSZICAR.load()
					data.append( self.OSZICAR.E[-1] )

				if extract == 'scale':
					self.POSCAR = POSCAR.POSCAR(f'{path}/{d}/CONTCAR')
					self.POSCAR.load()
					data.append( self.POSCAR.scale )		

				if extract == 'volumen':
					self.POSCAR = POSCAR.POSCAR(f'{path}/{d}/CONTCAR')
					self.POSCAR.load()
					data.append( self.POSCAR.volumen() )	

				if extract == 'ENCUT':
					self.INCAR = INCAR.INCAR(f'{path}/{d}/INCAR')	
					self.INCAR.load()
					data.append( self.INCAR.ENCUT )		

				if extract == 'KPOINTS':
					self.KPOINTS = KPOINTS.KPOINTS(f'{path}/{d}/KPOINTS')	
					self.KPOINTS.load()
					print( self.KPOINTS.subdivisions )
					data.append( self.KPOINTS.subdivisions[0] )		

		return data

slab = SLAB()
path = '/home/akaris/Documents/code/VASP/v4.7/files/bulk_optimization/Pt/initial_files/parametros/lattice_optimization_FCC_500_252525_LDA'
Xdata = slab.analysis(path=path, extract='energy')
#Ydata = slab.analysis(path=path, extract='ENCUT')
Ydata = slab.analysis(path=path, extract='volumen')
plt.plot(Ydata, Xdata, 'o', ms=2, color='r')

slab = SLAB()
path = '/home/akaris/Documents/code/VASP/v4.7/files/bulk_optimization/Pt/initial_files/parametros/lattice_optimization_BCC_500_252525_LDA'
Xdata = slab.analysis(path=path, extract='energy')
#Ydata = slab.analysis(path=path, extract='ENCUT')
Ydata = slab.analysis(path=path, extract='volumen')
plt.plot(Ydata, Xdata, 'o', ms=2, color='g')

slab = SLAB()
path = '/home/akaris/Documents/code/VASP/v4.7/files/bulk_optimization/Pt/initial_files/parametros/lattice_optimization_BCC_500_252525B'
Xdata = slab.analysis(path=path, extract='energy')
#Ydata = slab.analysis(path=path, extract='ENCUT')
Ydata = slab.analysis(path=path, extract='volumen')
plt.plot(Ydata, Xdata, 'o', ms=2, color='g')


slab = SLAB()
path = '/home/akaris/Documents/code/VASP/v4.7/files/bulk_optimization/Pt/initial_files/parametros/lattice_optimization_FCC_500_252525'
Xdata = slab.analysis(path=path, extract='energy')
#Ydata = slab.analysis(path=path, extract='ENCUT')
Ydata = slab.analysis(path=path, extract='volumen')
plt.plot(Ydata, Xdata, 'o', ms=2, color='r')


plt.show()
'''

slab = SLAB()
path = '/home/akaris/Documents/code/VASP/v4.7/files/bulk_optimization/Pt/initial_files/parametros/lattice_optimization_BCC_500_252525'
Xdata = slab.analysis(path=path, extract='energy')
#Ydata = slab.analysis(path=path, extract='ENCUT')
Ydata = slab.analysis(path=path, extract='volumen')
plt.plot(Ydata, Xdata, 'o', ms=2)


slab = SLAB()
path = '/home/akaris/Documents/code/VASP/v4.7/files/bulk_optimization/Pt/initial_files/parametros/lattice_optimization_BCC_500_252525B'
Xdata = slab.analysis(path=path, extract='energy')
#Ydata = slab.analysis(path=path, extract='ENCUT')
Ydata = slab.analysis(path=path, extract='volumen')
plt.plot(Ydata, Xdata, 'o', ms=2)


slab = SLAB()
path = '/home/akaris/Documents/code/VASP/v4.7/files/bulk_optimization/Pt/initial_files/parametros/lattice_optimization_HCP_252525_500'
Xdata = slab.analysis(path=path, extract='energy')
#Ydata = slab.analysis(path=path, extract='ENCUT')
Ydata = slab.analysis(path=path, extract='volumen')
plt.plot(Ydata, np.array(Xdata)/2, 'o', ms=2)


slab = SLAB()
path = '/home/akaris/Documents/code/VASP/v4.7/files/bulk_optimization/Pt/initial_files/parametros/lattice_optimization_SC_252525_500'
Xdata = slab.analysis(path=path, extract='energy')
#Ydata = slab.analysis(path=path, extract='ENCUT')
Ydata = slab.analysis(path=path, extract='volumen')
plt.plot(Ydata, np.array(Xdata)/2, 'o', ms=2)

slab = SLAB()
path = '/home/akaris/Documents/code/VASP/v4.7/files/bulk_optimization/Pt/initial_files/parametros/lattice_optimization_SC_252525_500B'
Xdata = slab.analysis(path=path, extract='energy')
#Ydata = slab.analysis(path=path, extract='ENCUT')
Ydata = slab.analysis(path=path, extract='volumen')
plt.plot(Ydata, np.array(Xdata)/2, 'o', ms=2)

slab = SLAB()
path = '/home/akaris/Documents/code/VASP/v4.7/files/bulk_optimization/Pt/initial_files/parametros/lattice_optimization_SC_252525_500C'
Xdata = slab.analysis(path=path, extract='energy')
#Ydata = slab.analysis(path=path, extract='ENCUT')
Ydata = slab.analysis(path=path, extract='volumen')
plt.plot(Ydata, np.array(Xdata)/2, 'o', ms=2)

plt.show()
asd
'''


if __name__ == "__main__":
	slab = SLAB()
	#slab.create( L=6,   atoms_dict = {'Pt':[(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)] }, 
	#					cell       = [[1,0,0],[0,1,0],[0,0,1]] )

	path = '/home/akaris/Documents/code/VASP/v4.7/files/bulk_optimization/Pt/initial_files'
	#slab.POSCAR.export('POSCAR') # export one POSCAR 
	slab.load_files(path=path) 
	print( slab.POSCAR )
	slab.variable_analysis( variable = 'lattice',
							values = [ 1.5 + float(n)*4.5/200 for n in range(200) ], 
							#values = [ [ 1+n, 1+n, 1+n, ] for n in range(35) ], 
							path=path )
