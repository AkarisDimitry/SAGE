print('Python 3.X - https://www.python.org/download/releases/3.0/')
###################################
# Step 1 || Load python libraries #
###################################
# *** warning supresion
import warnings
warnings.filterwarnings("ignore")

# *** warning supresion
import warnings; warnings.filterwarnings("ignore")

# *** python libraries
try:
	import itertools, operator, logging, time, copy, pickle, datetime, os, sys, argparse
	#os.chmod('.', 777)
except:  print('WARNING :: DATA.import_libraries() :: can not import itertools, operator, logging, time, copy, pickle or os')

# *** numpy libraries
try:
	import numpy as np
except: print('WARNING :: DATA.import_libraries() :: can not import numpy ')

# *** matplotlib libraries 
try:
	import matplotlib.pyplot as plt
	import matplotlib.axes as ax
	import matplotlib.patches as patches
except:	print('WARNING :: Set.import_libraries() :: can not import matplotlib ')

try:	from src import POSCAR
except:	
	try: import POSCAR as POSCAR
	except: print('WARNING :: Set.import_libraries() :: can not import POSCAR ')

try:	from src import DOSCAR
except:	
	try: import DOSCAR as DOSCAR
	except: print('WARNING :: Set.import_libraries() :: can not import DOSCAR ')

try:	from src import OUTCAR
except:	
	try: import OUTCAR as OUTCAR
	except: print('WARNING :: Set.import_libraries() :: can not import OUTCAR ')

try:	from src import OSZICAR
except:	
	try: import OSZICAR as OSZICAR
	except: print('WARNING :: Set.import_libraries() :: can not import OSZICAR ')

try:	from src import Data
except:	
	try: import Data as Data
	except: print('WARNING :: Set.import_libraries() :: can not import Data ')

try:	from src import System
except:	
	try: import System as System
	except: print('WARNING :: Set.import_libraries() :: can not import System ')

try:	from src import ORR
except:	
	try: import ORR as ORR
	except: print('WARNING :: Set.import_libraries() :: can not import ORR ')

try:	from src import Logs
except:	
	try: import Logs as Logs
	except: print('WARNING :: Set.import_libraries() :: can not import ORR ')

'''
try:
	#from scipy.signal import savgol_filter

	# NON linear models #
	from sklearn.neural_network  import MLPRegressor
	from sklearn.datasets 		 import make_regression
	from sklearn.model_selection import train_test_split
	from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit
	# linear models #
	from sklearn.decomposition import PCA
	from sklearn.preprocessing import StandardScaler
	from sklearn import linear_model
	from sklearn.model_selection import cross_val_predict
	from sklearn.metrics import mean_squared_error, r2_score

except:	print('ERROR :: DATA.import_libraries() :: can not import sklearn')
'''

try:
	from ase import Atoms
	from ase.visualize import view
except:	print('ERROR :: DATA.import_libraries() :: can not import ase ')

try:
	from io import StringIO
	import cProfile
	import pstats
except: print('ERROR :: DATA.import_libraries() :: can not import profiler tools ')

	# *****************************************************************************************************************************************************************
	# * === EMBEDDING === EMBEDDING === EMBEDDING === EMBEDDING === EMBEDDING === EMBEDDING === EMBEDDING === EMBEDDING === EMBEDDING === EMBEDDING === EMBEDDING === *
	# *****************************************************************************************************************************************************************
class EMBEDDING(object):
	def __init__(self, 	name:str=None, description=None, 
						path_list:list=None, name_list:list=None, ):
		self.name = name
		self.description = description

		#if action == 'load': return load_emb(**kwargs)
		#if action == 'make': return make_emb(**kwargs)
		#if action == 'save': return save_emb(**kwargs)

	def save(path, ff_X, ff_Y):
		if not os.path.isdir(path):  os.makedirs(path)

		for atom, ff_x in ff_X.items():
			if not os.path.isdir(f'{path}/{atom}'):  os.makedirs(f'{path}/{atom}')
			with open(f'{path}/{atom}/ff_X.dat', "ab") as file:
				np.savetxt(file, ff_x )

		for atom, ff_y in ff_Y.items():
			with open(f'{path}/{atom}/ff_Y.dat', "ab") as file:
				np.savetxt(file, ff_y )

		return None

	def load(path, v=True):
		print(path)
		Xembedding_dict = {}
		Yembedding_dict = {}
		if v: print('Loading embedding ...')
		
		for i, subfolder in enumerate(os.listdir(path)):
			t1 = time.time()
			Xembedding_dict[str(subfolder)] = np.loadtxt(f'{path}/{subfolder}/ff_X.dat' )
			Yembedding_dict[str(subfolder)] = np.loadtxt(f'{path}/{subfolder}/ff_Y.dat' )
			if v: print(f'\t ({i}) Load atom: {subfolder} \t Samples: {Xembedding_dict[str(subfolder)].shape[0]} \t Dimention {Xembedding_dict[str(subfolder)].shape[1]} \t ({time.time()-t1}s) ')

		return Xembedding_dict, Yembedding_dict

	def make(embedding_parameters=None, path=None, v=True, save=True):
		# === Get data from OUTCAR === #
		embedding_parameters = {'partition':400, 'min distance':0.5, 'max distance':7.0, 'range distance':0.001} if embedding_parameters == None else embedding_parameters
				
		for dataset_n, (key_dataset, dataset) in enumerate(self.set.items()):
			if v: print(' Getting embeding from  {}'.format(key_dataset) )
			result_list = dataset.get_embeddings(embedding_parameters=embedding_parameters, path=path, 
												processing_pool={'multiprocessing':True, 'cores':'check'}, save=False)

			if save:
				for result in result_list:	
					save_emb(path, result['ff_X'], result['ff_Y'])

		return load_emb(path, v=v)

class FORCEFIELD(object):
	def forcefield(self, embedding:str ='make', embedding_parameters=None, 
						 forcefield:str='make', MLPR=None, 
						 path=None,  save=True, v=True):
		self.embedding 	= embedding
		self.embedding_parameters = embedding_parameters
		self.forcefield = forcefield
		self.MLPR 		= MLPR
		self.path 		= path
		self.save 		= save
		self.verbose 	= verbose

		'''
		dataset_path = '/home/akaris/Documents/code/VASP/v3.5/files/dataset/force_trainning/dataset_ff.pkl'
		ff_path = '/home/akaris/Documents/code/VASP/v3.5/files/dataset/force_trainning/force_field/02'
		dataset = Set()
		dataset.load_data( filename=dataset_path )
		MLPR = dataset.forcefield( path=ff_path, embedding='None', forcefield='load' )
		'''

		def load_forcefield(self, path=None, v=True, save=True):
			MLPR = {} 

			for i, subfolder in enumerate(os.listdir(path)):
				t1 = time.time()

				try:
					with open(f'{path}/{subfolder}/regrNN', 'rb') as file:
						MLPR[str(subfolder)] = pickle.load(file)
						if v: print(f'Reading atom: {subfolder} \t time {time.time()-t1}')

				except:
					if v: print(f'Can not find file: *regrNN* in path {path}/subfolder/')

			return MLPR

		def save_forcefield(self, path=None , regr=None):
			with open(f'{path}/regrNN', 'wb') as file:
				pickle.dump(regr, file)
		
		def train_forcefield(self, ff_X, ff_Y, MLPR=None, v=True, save=True ):
			MLPR = {} if type(MLPR) != dict else MLPR
			for atom, data in ff_X.items():
				# ========== Filter data ========== #
				ffx_atom_train, ffy_atom_train = ff_X[str(atom)], ff_Y[str(atom)]
				filters = np.sum(np.abs( ffy_atom_train ),axis=1) < 0.5

				ffx_atom_train = ffx_atom_train[filters,:]
				ffy_atom_train = ffy_atom_train[filters,:]

				# ========== Train model NN ========== #
				if type(MLPR) == dict and str(atom) in MLPR:
					regr_atom = MLPR[atom].partial_fit(ffx_atom_train, ffy_atom_train)
				else:
					regr_atom = MLPRegressor(random_state=3, activation='tanh', #early_stopping=True, validation_fraction=0.10,
									hidden_layer_sizes=(30, 30, 30, 30),max_iter=1000, #learning_rate='adaptive', warm_start=True,
									#hidden_layer_sizes=(  400, 540, 540, 400,  ),max_iter=5000,
									momentum=0.7, verbose=True, n_iter_no_change=180, 
									early_stopping=True, validation_fraction=0.1).fit(ffx_atom_train, ffy_atom_train)

				MLPR[str(atom)] = regr_atom

				# ========== SAVE model NN ========== #
				if not os.path.isdir(f'{path}/{atom}'):  os.makedirs(f'{path}/{atom}')
				if save: save_forcefield( path=f'{path}/{atom}' , regr=regr_atom)

			return MLPR

		def validate_forcefield(self, path, MLPR, ff_X, ff_Y, v=True, save=True ):
			for key, value in ff_X.items():
				filters = np.sum(np.abs( ff_Y[key] ),axis=1) < 1
				value = value[filters,:]
				ff_Y[key] = ff_Y[key][filters,:]

				ff_y = MLPR[key].predict( value )
				plt.plot( ff_Y[key][:,0], ff_y[:,0], 'o', c='r', ) 
				plt.plot( ff_Y[key][:,1], ff_y[:,1], 'o', c='g', ) 
				plt.plot( ff_Y[key][:,2], ff_y[:,2], 'o', c='b', ) 
				plt.title(f'{key}')
				plt.show()

		def initialize():
			# === Embedding === #
			if   embedding == 'make': ff_X, ff_Y = self.embedding('make', embedding_parameters=embedding_parameters, path=path, v=v, save=save)
			elif embedding == 'load': ff_X, ff_Y = self.embedding('load', path=path, v=v, )
			else: ff_X, ff_Y = None, None

			# === Multi-layer Perceptron regressor === #
			if   forcefield == 'make': MLPR = train_forcefield( ff_X, ff_Y, MLPR, v=v, save=save )
			elif forcefield == 'load': MLPR = load_forcefield( path, v=v, save=save )
			elif forcefield == 'validate': 
				MLPR=load_forcefield( path, v=v, save=save )
				validate_forcefield( path, MLPR, ff_X, ff_Y, v=v, save=save )
			else: MLPR = {}

			if save: self.ff_MLPR = MLPR

			return MLPR
