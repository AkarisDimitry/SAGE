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

class Set(object):
	def __init__(self, 	name=None, description=None, 
						path_list=None, name_list=None, ):
		self.name = name
		self.description = description

		self.path_list = path_list
		self.name_list = name_list

		self.set = {}

	# *****************************************************************************************************************************************************************
	# * === LOAD/SAVE === LOAD/SAVE === LOAD/SAVE === LOAD/SAVE === LOAD/SAVE === LOAD/SAVE === LOAD/SAVE === LOAD/SAVE === LOAD/SAVE === LOAD/SAVE === LOAD/SAVE === *
	# *****************************************************************************************************************************************************************
	@Logs.LogDecorator()	
	def read_recursive(self, path:str=None, output:str=None, files:list=['all'], query:dict=None,
						save:bool=True, verbosity:bool=True, append:bool=True, save_format:dict={'pickle'}) -> dict:									
		# ----------------------------------------------------------------------------------		
		'''
		# READ dataset #t
		quert		:: 		dict 		:: 	dicttionary with the query
		file 		::		str 		::  file name to save data
		dataset 	:: 		obj 		:: 	dictionnary  that conteins all path to data set elements
		save 		:; 		bool 		:: 	Indicates if its nesesary to save 
		verbosity ::		bool 		:: 	Indicates if its nesesary print information during the processes 
		# ----------------------------------------------------------------------------------				
		1.       Read the dataset.
		2.       If verbosity is set to True, print information during the process.
		3.        For each set in the dataset, create a Data object.
		4.        Load the system for the set into memory.
		5.        Do an ORR on the data.
		6.        Output the summary of the data.
		7.        If save is set to True, save the data to a file.
		# ----------------------------------------------------------------------------------	
		query example: 

		query:dict={
		'ORR'		:[
				'overpotencial_OER_4e','overpotencial_ORR_4e', 
				'G1_OER'  ,	'G2_OER'  ,	'G3_OER' ,	'G4_OER' ,
				'G1_ORR'  ,	'G2_ORR'  ,	'G3_ORR' ,	'G4_ORR' ,
				'Eabs_OOH',	'Eabs_O'  ,	'Eabs_OH',
				'Gabs_OH' ,	'Gabs_OOH',	'Gabs_O'	], 
		'E'			: [], 
		'distances'	: [0, 1] 
		}, 	
		'''
		# ----------------------------------------------------------------------------------	
		query = query if not query is None else {}
		query_dict = {} if not query is None else None
		v 		 = verbosity
		output 	 = output 	if type(output) 	== str 	else 'dataset_save_state' 
		path  	 = path 	if type(path)		== str else '.'
		files 	 = ['POSCAR', 'CONTCAR', 'OUTCAR', 'OSZICAR', 'INCAR'] if files == 'all' else files
		self.set = {}		if not type(self.set) is dict or not append  else self.set
		
		#walk_dir = os.path.abspath(walk_dir)
		if v: print(f'>> Walk dir : {path} ')
		system_num = 0
		for root, root_subdirs, root_files in os.walk(path):
			#for data_name, data_dict in dataset.items():
			# LOAD >> make data obj and read Data
			if np.sum([1 if f in root_files else 0 for f in files]) > 0:
				if v: print('-'*40+f'\n >>  Root | {root} {root_files} TOTAL {system_num}'+f'\n')
				data = Data.Data()
				data.load(data={f'system_{system_num}':root}, files=files, v=v)
				system_num += 1

				# STORE >> store query data in self.set
				try:	self.set[root] = data
				except: print('ERROR :: Set.read_data() :: can not merge data ')

		# SAVE >> save data into hard drive
		if save: 	self.save_set(filename=f'{output}', dataset=self.set, save_format=save_format, query_dict=query_dict)
		return True

	def load_recursive(self, filename:str=None, 
							save:bool=True, verbosity:bool=True, append:bool=True, save_format:dict={'pickle'}) -> dict:									
		# ----------------------------------------------------------------------------------		
		'''
		# READ dataset #t
		quert		:: 		dict 		:: 	dicttionary with the query
		file 		::		str 		::  file name to save data
		dataset 	:: 		obj 		:: 	dictionnary  that conteins all path to data set elements
		save 		:; 		bool 		:: 	Indicates if its nesesary to save 
		verbosity ::		bool 		:: 	Indicates if its nesesary print information during the processes 
		# ----------------------------------------------------------------------------------				
		1.       Read the dataset.
		2.       If verbosity is set to True, print information during the process.
		3.        For each set in the dataset, create a Data object.
		4.        Load the system for the set into memory.
		5.        Do an ORR on the data.
		6.        Output the summary of the data.
		7.        If save is set to True, save the data to a file.
		# ----------------------------------------------------------------------------------	
		query example: 

		query:dict={
		'ORR'		:[
				'overpotencial_OER_4e','overpotencial_ORR_4e', 
				'G1_OER'  ,	'G2_OER'  ,	'G3_OER' ,	'G4_OER' ,
				'G1_ORR'  ,	'G2_ORR'  ,	'G3_ORR' ,	'G4_ORR' ,
				'Eabs_OOH',	'Eabs_O'  ,	'Eabs_OH',
				'Gabs_OH' ,	'Gabs_OOH',	'Gabs_O'	], 
		'E'			: [], 
		'distances'	: [0, 1] 
		}, 	
		'''
		# ----------------------------------------------------------------------------------	
		dataset = pickle.load( open( filename, "rb" ), encoding='latin1') # encoding='bytes'
		self.__dict__ = dataset.__dict__.copy() 
		print(f' >> Successfully read {filename}\n')
		if verbosity:
			for i, (key, item) in enumerate(self.set.items()):
				for j, (key2, item2) in enumerate(item.system.items()):
					print(f'({i:>5}) {key:<130} >> {item2.loaded}')
					#try: print( item2.OSZICAR.E[-1] )
					#except: pass

		return dataset


	@Logs.LogDecorator()	
	def read_data(self, dataset:dict=None, output:str=None, files:list=['all'], query:dict=None,
						save:bool=True, verbosity:bool=True, append:bool=True, save_format:dict={'pickle'}) -> dict:									
		# ----------------------------------------------------------------------------------		
		'''
		# READ dataset #t
		quert		:: 		dict 		:: 	dicttionary with the query
		file 		::		str 		::  file name to save data
		dataset 	:: 		obj 		:: 	dictionnary  that conteins all path to data set elements
		save 		:; 		bool 		:: 	Indicates if its nesesary to save 
		verbosity ::		bool 		:: 	Indicates if its nesesary print information during the processes 
		# ----------------------------------------------------------------------------------				
		1.       Read the dataset.
		2.       If verbosity is set to True, print information during the process.
		3.        For each set in the dataset, create a Data object.
		4.        Load the system for the set into memory.
		5.        Do an ORR on the data.
		6.        Output the summary of the data.
		7.        If save is set to True, save the data to a file.
		# ----------------------------------------------------------------------------------	
		query example: 

		query:dict={
		'ORR'		:[
				'overpotencial_OER_4e','overpotencial_ORR_4e', 
				'G1_OER'  ,	'G2_OER'  ,	'G3_OER' ,	'G4_OER' ,
				'G1_ORR'  ,	'G2_ORR'  ,	'G3_ORR' ,	'G4_ORR' ,
				'Eabs_OOH',	'Eabs_O'  ,	'Eabs_OH',
				'Gabs_OH' ,	'Gabs_OOH',	'Gabs_O'	], 
		'E'			: [], 
		'distances'	: [0, 1] 
		}, 	
		'''
		# ----------------------------------------------------------------------------------	
		query = query if not query is None else {}
		query_dict = {} if not query is None else None
		v 		 = verbosity
		output 	 = output 	if type(output) 	== str 	else 'dataset_save_state' 
		dataset  = dataset 	if type(dataset)	== dict else self.dataset
		if not type(dataset) == dict: 	print('ERROR : code000 READ.read_data() : need dir/dataset ')

		self.set = {}		if not type(self.set) is dict or not append  else self.set

		for data_name, data_dict in dataset.items():
			# LOAD >> make data obj and read Data
			if v: print('-'*40+f'\n >>  {data_name} '+f'\n')
			data = Data.Data()
			data.load(data=data_dict, files=files, v=v)

			# EVALUATE >> evaluate ORR from data
			try:
				data.ORR()
				if v: data.summary( )
			except: pass

			# GET >> get query_dict data
			if not query is None: 
				query_dict[data_name] = {}
				if 'E'    in query:	query_dict[data_name]['E']    = {system_name : system_obj.OSZICAR.E[-1] for system_name, system_obj in data.system.items() }
				if 'ORR'  in query:	query_dict[data_name]['ORR']  = {query_ORR : data.ORR_analysis.ORR[query_ORR] for query_ORR in query['ORR'] }
				if 'PDOS' in query:	query_dict[data_name]['PDOS'] = {system_name : system_obj.DOSCAR.E[-1] for system_name, system_obj in data.system.items() }
				if v: print(query_dict)

			# STORE >> store query data in self.set
			try:self.set[data_name] = data
			except: print('ERROR :: Set.read_data() :: can not merge data ')

		# SAVE >> save data into hard drive
		if save: self.save_set(filename=f'{output}', dataset=self.set, save_format=save_format, query_dict=query_dict)
		return True

	@Logs.LogDecorator()
	def save_set(self, filename:str=None, dataset=None, 
				save_format:dict={'pickle', 'light'}, query_dict:dict=None) -> bool:
		'''
		1.      It determines whether the dataset is a dict or a list. If it's a list, it loops through the dataset and creates an object for each set in the dataset.
		2.      It creates a dataset_matrix object that will hold the data for each set in the dataset.
		3.      It loops through the dataset and creates an object for each set in the dataset.
		4.      It passes the objects created in step 3 to the pickle.dump() method.
		5.      The pickle.dump() method dumps the objects into a file.
		6.      The file handler is opened and the data is dumped into it.
		7.      The file is closed.

		You can use this class to load data into a Python program.
		'''
		dataset = dataset if type(dataset) == dict else self.set

		if 'light' in save_format:
			filename_light = f'{filename}.dat' if type(filename) is str else 'output_lightset.dat'

			query_file = open(filename_light, 'w') 
			for data_name, data_item in query_dict.items():
				query_file.write(f'{data_name:<.20s}\t')
				if 'E' in data_item:	
					for system_name, system_data in data_item['E'].items():
						query_file.write(f'{system_data:<.4f}\t')
				
				if 'ORR' in data_item:	
					for system_name, system_data in data_item['ORR'].items():
						query_file.write(f'{system_data:<.4f}\t')
				
				query_file.write('\n')
			query_file.close()

		if 'pickle' in save_format:
			filename = f'{filename}.pkl' if type(filename) is str else 'output_set.pkl'
			filehandler = open(filename, 'wb') 
			pickle.dump(dataset, filehandler)
			filehandler.close()

		return True

	def load_data(self, filename:str, verbosity:bool=True) -> dict:
		dataset = pickle.load( open( filename, "rb" ), encoding='latin1') # encoding='bytes'
		self.__dict__ = dataset.__dict__.copy() 
		print(f' >> Successfully read {filename}\n')
		return dataset

	def get_data(self, dataset=None, file:str=None, query:dict={'E':['total'], 'distances':[0, 1] }, verbosity:bool=True, save:bool=True) -> dict:
		# ----------------------------------------------------------------------------------
		# This function collect data from dataset and make tables from it
		# quert		:: 		dict 		:: 	dicttionary with the query
		# file 		::		str 		:: 	file name to save data
		# dataset 	:: 		obj 		:: 	OBJ from data will be read 
		# verbosity ::		bool 		:: 	activate/desactiuvate verbosity	
		# save 		:; 		bool 		:: define if data will be save
		# ----------------------------------------------------------------------------------
			
		# ---- Genral asignation ---- # 
		v 		= verbosity
		file 	= output  if type(file) 	is str 	else 'dataset_save_state' 
		dataset = dataset if type(dataset)	is dict else self.dataset
		if not type(dataset) == dict: 	print('ERROR : code000 READ.read_data() : need dir/dataset ')

		for set_name in dataset:
			if v: print('='*70+f'\n>>  * {set_name} '+f'\n    |  ')
			sub_system = dataset[set_name]
			data = Data.Data()

			for sys_name in sub_system: 
				#if v: print(f'    *=== {set_name}+{sys_name}')
				sys_path = sub_system[sys_name]

				data.load_system(	name  = f'{set_name}_{sys_name}',
									path  = sys_path, 
									files = ['CONTCAR', 'OSZICAR'],
									v     = v
								)

				system = data.system[f'{set_name}_{sys_name}']

				try:
					data_vec = []
					if   'E'	 	 in query: data_vec.append( '{:.3f}'.format(system.OSZICAR.E[-1]) )
					elif 'distances' in query: data_vec.append( '{:.3f}'.format(system.CONTCAR.periodic_distance(atoms=query['distances'])) )

				except: pass

			self.set[str(set_name)] = data

			if save: np.savetxt(fname=f'{file}_{set_name}.dat',X=data_vec)

		return True
		'''
		dataset = pickle.load( open( filename, "rb" ), encoding='latin1') # encoding='bytes'
		self.__dict__ = dataset.__dict__.copy() 
		print(f' >> Successfully read {filename}\n')
		return dataset
		'''

	def summary(self, feature:dict={'ORR':True} ) -> bool:
		for set_n, (key_data, data) in enumerate(self.set.items()):
			print( ' {1} System {0} {1}'.format(key_data, '*'*12) )
			data.summary( feature={'ORR':True} )	

		return True

	# ****************************************************************************************************************************************************************
	# * === PLOT === PLOT === PLOT === PLOT === PLOT === PLOT === PLOT === PLOT === PLOT === PLOT === PLOT === PLOT === PLOT === PLOT === PLOT === PLOT === PLOT === *
	# ****************************************************************************************************************************************************************
	def plot_angle(self, atom1, atom2, atom3, atom4=None, ax=None, save=False, path='./no_name.png' ):
		if ax==None:	fig, ax = plt.subplots()
		else:			fig = ax.get_figure()

		angles, cell, E = [], [], []
		for set_n, (key_data, data) in enumerate(self.set.items()):
			for system_n, (key_system, system) in enumerate(data.system.items()):
				
				if not system.CONTCAR is None:

					if not system.OSZICAR is None and not system.CONTCAR is None:
						#d1 = system.CONTCAR.get_dihedric(atom1, atom2, atom3, atom4)*180/3.1415
						#angles.append( d1 )
						try:
							E.append(system.OSZICAR.E[-1])
							cell.append([system.CONTCAR.cell[0][0], system.CONTCAR.cell[1][1]])
						except: pass
		#ax.plot(cell, angles, 'o' )
		ax.plot(cell, E, 'o' )
		cell = np.array(cell)
		ax.set_title('Angles')
		ax.set_ylabel('Angle (randians)')
		ax.set_xlabel('System')
		plt.show()
		if save:
			fig.savefig(f'{path}' , dpi=100, pad_inches=0.1, 
						bbox_inches='tight', horizontalalignment='right') 
			np.savetxt('data.data', np.array([cell[:,0], cell[:,1], E]).T )
			
		return cell, angles

	def extract_features(self, feature=None, v=True, filters='None'):
		# extract specific features 
		# ----------------------------------------------------------------------------------------

		# ----------------------------------------------------------------------------------------
		# feature 	::	dict 	:: feature to extract 
		# v 		::  BOOL 	:: vervosity eg: True
		# filter 	::  STR 	:: filter on features space

		'''
		feature = { 'ORR' : ['overpotencial_ORR_4e']
					'PDOS': {	'config' : {'start':-1.0, 'end':1.0, 'point':500}
								'atoms'  : {'name':['Co', 'Fe']}} ,  or {'closest':'_*O'}
								'orbital': ['d']  } 
					'magnetization'	: {'atoms'  : {'name':['Co', 'Fe']}} 
					'charge'		: {'atoms'  : {'name':['Co', 'Fe']}}
					}	
		'''
		# -- set data storage -- #
		feature_ext = {} # here we extract all data
		if v: print('*'*10 + '\nExtracting features\n'+'*'*10) # verbosity 

		# *********** ITERATION data *********** # # *********** ITERATION data *********** # # *********** ITERATION data *********** # # *********** ITERATION data *********** # 
		for set_n, (key_data, data) in enumerate(self.set.items()):
			key_data = key_data.split('\n')[-1]
			if v: print(' [{1}] reading system :: {0} :: '.format(str(key_data), str(set_n)) ) # verbosity 

# ------- ORR ------- # # ------- ORR ------- # # ------- ORR ------- # # ------- ORR ------- # # ------- ORR ------- # # ------- ORR ------- # # ------- ORR ------- #
			if 'ORR' in feature:
				try:
					ORR = data.ORR()
					if not key_data in feature_ext: feature_ext[key_data] = {}
					feature_ext[key_data]['ORR'] = [ ORR.ORR[orr_feature] for orr_feature in feature['ORR'] ]
				except:
					feature_ext[key_data]['ORR'] = [None]

				if v: 
					try:
						print(' [{1}] :: {0} :: evaluating ORR parameters.'.format(str(key_data), str(set_n)) ) # verbosity 				
						for orr_feature in feature['ORR']:	print(' [{3}] :: {0} :: ORR :: {1} {2} '.format(str(key_data), str(orr_feature), str(ORR.ORR[orr_feature]), str(set_n), ) ) # verbosity xxx
					except:
						print('WARNING :: Set.extract_features() :: can NOT print OOR summary')
# ------- ORR ------- # # ------- ORR ------- # # ------- ORR ------- # # ------- ORR ------- # # ------- ORR ------- # # ------- ORR ------- # # ------- ORR ------- #
			
			# *********** ITERATION system *********** # # *********** ITERATION system *********** # # *********** ITERATION system *********** # # *********** ITERATION system *********** #
			for system_n, (key_system, system) in enumerate(data.system.items()):
				key_system = key_system.split('\n')[-1]
				if v: print(' [{2}.{3}] :: {0} :: {1} ::'.format(str(key_data), str(key_system), str(set_n), str(system_n) )) # verbosity 
				
# ------- DOSCAR/ ------- # # ------- DOSCAR/ ------- # # ------- DOSCAR/ ------- # # ------- DOSCAR/ ------- # # ------- DOSCAR/ ------- # # ------- DOSCAR/ ------- #
				if not system.DOSCAR is None and 'PDOS' in feature:
					if v: print(' [{2}.{3}] :: {0} :: {1} :: PDOS'.format(str(key_data), str(key_system), str(set_n), str(system_n)  )) # verbosity 
					DOSCAR_feature = []

					atom_positions = []
					if 'name' in feature['PDOS']['atoms'] and not system.OUTCAR is None:
						atom_positions = []
						for a in feature['PDOS']['atoms']['name']:
							atom_positions += system.OUTCAR.get_position(a)

					if 'number' in feature['PDOS']['atoms']:
						atom_positions += [num for num in feature['PDOS']['atoms']['number']]

					atom_positions = [1] if 1 in atom_positions else atom_positions

					for orbital in feature['PDOS']['orbital']:
						config = feature['PDOS']['config']

						for ap in atom_positions:
							DOSCAR_feature.append( system.DOSCAR.cut( 	ion=ap, orbital=orbital, 
																		start= 	config['start'], 
																		end=	config['end']  ,
																		point=	config['point'], ) )
				else: DOSCAR_feature = None
# ------- /DOSCAR ------- # # ------- /DOSCAR ------- # # ------- /DOSCAR ------- # # ------- /DOSCAR ------- # # ------- /DOSCAR ------- # # ------- /DOSCAR ------- #

# ------- relevant_distances  ------- # # ------- relevant_distances  ------- # # ------- relevant_distances  ------- # # ------- relevant_distances  ------- # 
				if not system.CONTCAR is None and 'relevant_distances' in feature:
					system.CONTCAR.relevant_distances(relevance=2, relevance_distance=2.5, save=True, v=v)

# ------- magnetization/  ------- # # ------- magnetization/  ------- # # ------- magnetization/  ------- # # ------- magnetization/  ------- # # ------- magnetization/  ------- #
				if not system.OUTCAR is None and 'magnetization' in feature:
					if 'name' in feature['magnetization']['atoms'] :
						atom_positions = []
						for a in feature['magnetization']['atoms']['name']:
							atom_positions += system.OUTCAR.get_position(a)

					if 'number' in feature['magnetization']['atoms']:
						atom_positions = [num for num in feature['PDOS']['atoms']['number']]

					magnetization_feature = system.OUTCAR.get_magnetization( atom_positions )
					if v: print(' [{2}.{3}] :: {0} :: {1} :: magnetization {4}'.format(str(key_data), str(key_system), str(set_n), str(system_n), magnetization_feature)) # verbosity 
				else: magnetization_feature = None
# ------- /magnetization  ------- # # ------- /magnetization  ------- # # ------- /magnetization  ------- # # ------- /magnetization  ------- # # ------- /magnetization  ------- #

# ------- charge/  ------- # # ------- charge/  ------- # # ------- charge/  ------- # # ------- charge/  ------- # # ------- charge/  ------- # # ------- charge/  ------- #
				if not system.OUTCAR is None and 'charge' in feature:

					if 'name' in feature['charge']['atoms'] :
						atom_positions = []
						for a in feature['charge']['atoms']['name']:
							atom_positions += system.OUTCAR.get_position(a)

					if 'number' in feature['charge']['atoms']:
						atom_positions = [num for num in feature['PDOS']['atoms']['number']]

					charge_feature = system.OUTCAR.get_charge( atom_positions )
					if v: print(' [{2}.{3}] :: {0} :: {1} :: charge {4}'.format(str(key_data), str(key_system), str(set_n), str(system_n), charge_feature)) # verbosity 
				else: charge_feature = None
# ------- /charge  ------- # # ------- /charge  ------- # # ------- /charge  ------- # # ------- /charge  ------- # # ------- /charge  ------- # # ------- /charge  ------- #
				if not key_data   in feature_ext: 			feature_ext[key_data] 				= {}
				if not key_system in feature_ext[key_data]: feature_ext[key_data][key_system] 	= {}

				if 'charge' 	   in feature:	feature_ext[key_data][key_system]['charge'] 		= charge_feature
				if 'magnetization' in feature:	feature_ext[key_data][key_system]['magnetization'] 	= magnetization_feature 
				if 'PDOS' 		   in feature:	feature_ext[key_data][key_system]['PDOS'] 			= DOSCAR_feature

		self.feature_ext = feature_ext
		return feature_ext 

	def feature_dic2array(self, dictionary, ):
		# transform dictionary to the corresponding arrays

		dictionary = dictionary if type(dictionary) == dict else self.feature_ext
		#dic_array = { key_data:[] for (key_data, data) in dictionary.items() }
		dic_array = {}
		list_name = [ key_label for (key_label, value_label) in dictionary.items() ]

		for (key_label, value_label) in dictionary.items():
			for (key_data, value_data) in value_label.items():
				for value in value_data:
					if key_data in dic_array:
						dic_array[key_data] = np.concatenate( (dic_array[key_data], np.array(value)) )
					else:
						dic_array[key_data] = np.array(value)

		for (key_data, data) in dic_array.items():
			dic_array[key_data] = np.array(data)

		return dic_array, list_name

	def get_dataset(self, save_path=None, features=None):
		# this function generates a complete dataset 
		# 	X 	[samples, features] 	

		features = {'PDOS': {	'config' : {'start':-5.0, 'end':3.0, 'point':500},
										'atoms'  : {'name':['Fe', 'Co', 'Cu', 'Mg']},  
										'orbital': [9,10,11,12,13,14,15,16,17,18,]  },

					'magnetization'	: {'atoms'  : {'name':['Fe', 'Co', 'Cu', 'Mg']},},
					'charge'		: {'atoms'  : {'name':['Fe', 'Co', 'Cu', 'Mg']},}, 

					'ORR' : [	'overpotencial_ORR_4e', 'Gabs_OOH', 'Eabs_OOH', 'Gabs_OH', 
								'Eabs_OH', 'Gabs_O', 'Eabs_O', 'G1_ORR', 'G2_ORR', 'G3_ORR', 'G4_ORR', 'limiting_step_ORR'],
							} if type(features) != dict else features

		features_ext = self.extract_features(feature=features, v=False )

		data = []#
		for key_data, value_data in features_ext.items():
			vec = [key_data]
			for key_system, value_system in value_data.items():
				if key_system == 'ORR' and 'ORR' in features: 
					vec.append( value_data['ORR'] )
				else:
					vec.append( key_system )
					if 'PDOS' 			in features: vec.append( value_system['PDOS'] )
					if 'magnetization' 	in features: vec.append( value_system['magnetization'] )
					if 'charge' 		in features: vec.append( value_system['charge'] )
			data.append(vec)

		return data

	@Logs.LogDecorator()
	def get_dataset_PDOS2OP(self, save_path=None, features:dict=None, export:str=None, save:bool=True) -> list: 

		features = {'PDOS': {	'config' : {'start':-5.0, 'end':5.0, 'point':500},
										'atoms'  : {'name':['Al', 'Bi', 'Ca_sv', 'Cd', 'Co', 'Cr', 'Cu', 'Fe', 'Ir', 'Mg', 'Mn', 'Mo', 'Ni', 'Pb', 'Pd', 'Pt', 'Rh', 'Ru', 'Sc', 'Tc', 'Ti', 'V', 'Zn']},  
										'orbital': [9,10,11,12, 13,14, 15,16,17,18,]  },
					'magnetization'	:   {'atoms'  : {'name':['Al', 'Bi', 'Ca_sv', 'Cd', 'Co', 'Cr', 'Cu', 'Fe', 'Ir', 'Mg', 'Mn', 'Mo', 'Ni', 'Pb', 'Pd', 'Pt', 'Rh', 'Ru', 'Sc', 'Tc', 'Ti', 'V', 'Zn']},},
					'charge'		: 	{'atoms'  : {'name':['Al', 'Bi', 'Ca_sv', 'Cd', 'Co', 'Cr', 'Cu', 'Fe', 'Ir', 'Mg', 'Mn', 'Mo', 'Ni', 'Pb', 'Pd', 'Pt', 'Rh', 'Ru', 'Sc', 'Tc', 'Ti', 'V', 'Zn']},},  #{'atoms'  : {'number': [0] },},

					'ORR' : [	'overpotencial_ORR_4e', 'Gabs_OOH', 'Eabs_OOH', 'Gabs_OH', 
								'Eabs_OH', 'Gabs_O', 'Eabs_O', 'G1_ORR', 'G2_ORR', 'G3_ORR', 'G4_ORR', 'limiting_step_ORR'],
							} if type(features) != dict else features

		features_ext = self.extract_features(feature=features, v=False )

		Xdata, Ydata, names = [], [], []
		for key_data, value_data in features_ext.items():
			Xvec, Yvec, Nvec = [], np.array([]), key_data
			for key_system, value_system in value_data.items():
				if key_system == 'ORR' and 'ORR' in features: 
					Xvec = value_data['ORR'] 
				elif key_system[-1] == '*':

					if 'magnetization' 	in features and not value_system['magnetization'] is None: 
						Yvec = np.concatenate( (Yvec, value_system['magnetization'][0]) )
					if 'charge' 		in features and not value_system['charge']		  is None: 
						Yvec = np.concatenate( (Yvec, value_system['charge'][0]) )
				elif 'PDOS' in key_system:
					if 'PDOS' 			in features: Yvec = np.concatenate( (Yvec, np.array(value_system['PDOS']).flatten()) ) 

			#print(Xvec, Yvec , Nvec)
			#if len(Xvec) > 1 and len(Yvec) > 50:
			try:
				Xdata.append(Xvec)
				Ydata.append(Yvec)
				names.append(Nvec)
			except:
				pass

		if save:
			self.Xdata = Xdata
			self.Ydata = Ydata
			self.names = names
		
		# === make and array === #
		Ydim = max([len(y)for y in Ydata])
		Xdim = max([len(x)for x in Xdata])
		Xdata_array = np.zeros((len(Xdata), Xdim))
		Ydata_array = np.zeros((len(Ydata), Ydim))
		for i, y in enumerate(Ydata):
			try: Xdata_array[i, :] = Xdata[i]
			except: pass
			try: 	Ydata_array[i, :] = Ydata[i]
			except: pass

		return Xdata_array, Ydata_array, np.array(names)

	def extract_recursive(self, path:str=None):
		# decompres data from self.load_recursive()
		path = path if not path is None else '.'

		def make_path(path):
			if not os.path.isdir(path):  	 			
				os.makedirs(path)
				return True

		for data_key, data in self.set.items():
			print(f' >> Making path :: {path}/{data_key}')
			make_path(f'{path}/{data_key}')
			for system_key, system in data.system.items():
				try:
					system.CONTCAR.export(f'{path}/{data_key}/CONTCAR') 
					system.OSZICAR.export(f'{path}/{data_key}/OSZICAR') 
				except: print(f' WARNNING :: fail at path {path}/{data_key}')
		
		return True

	# Define the 'get_occupancy' method with default parameters for ion, orbital, save, and path.
	def get_occupancy(self, ion:list=[1], orbital:list=['all'], save:bool=False, path:str='.'):
		# Initialize an empty list to store occupancy data.
		ocupation_data = []

		for data_key, data in self.set.items():
			for system_key, system in data.system.items():
				# Check if the DOSCAR attribute of the 'system' object is not None, and if 'PDOS' is a substring of 'system_key'.
				if not system.DOSCAR is None and 'PDOS' in system_key:
					print(f'>> Reading ocupation : {data_key} - {system_key}')
					ocupation_dict = system.DOSCAR.get_occupancy(ion=ion, orbital=orbital, save=False, v=False)
					vec = [f'{data_key}-{system_key}']
					for atom, atom_ocupation in ocupation_dict.items():
						for orb, ocupation in atom_ocupation.items():
							# Create a list 'vec' with a single string element containing 'data_key' and 'system_key'.
							vec += [ocupation['ocupated']['up']-ocupation['ocupated']['down'], ocupation['Unocupated']['up']-ocupation['Unocupated']['down']]

					ocupation_data.append(vec)
		ocupation_data = np.array(ocupation_data)
		
		if save: np.savetxt( f'{path}', ocupation_data, delimiter=" ", fmt="%s")
		self.ocupation_data = ocupation_data
		
		return True


	def get_center_of_band(self, ion:list=[1], orbital:list=['all'], save:bool=False, path:str='.'):

		center_of_band_data = []

		for data_key, data in self.set.items():
			for system_key, system in data.system.items():
				if not system.DOSCAR is None and 'PDOS' in system_key:
					print(f'>> Reading ocupation : {data_key} - {system_key}')
					center_of_band = system.DOSCAR.get_center_of_band(ion=ion, orbital=orbital, save=False, v=False)
					center_of_band_data.append( [f'{data_key}-{system_key}', center_of_band] )
		center_of_band_data = np.array(center_of_band_data)
		
		if save: np.savetxt( f'{path}', center_of_band_data, delimiter=" ", fmt="%s")
		self.center_of_band_data = center_of_band_data
		
		return True

# ----------- PROFILER ----------- #
class Profiler(object): 

    def __init__(self, enabled=False, contextstr=None, fraction=1.0,
                 sort_by='time', parent=None, logger=None):
        self.enabled = enabled

        self.contextstr = contextstr or str(self.__class__)

        if fraction > 1.0 or fraction < 0.0:
            fraction = 1.0

        self.fraction = fraction
        self.sort_by = sort_by #cumulative

        self.parent = parent
        self.logger = logger

        self.stream = StringIO()
        self.profiler = cProfile.Profile()

    def __enter__(self, *args):

        if not self.enabled:
            return self

        # Start profiling.
        self.stream.write("\nprofile: {}: enter\n".format(self.contextstr))
        self.profiler.enable()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        if not self.enabled:
            return False

        self.profiler.disable()

        sort_by = self.sort_by
        ps = pstats.Stats(self.profiler, stream=self.stream).sort_stats(sort_by)
        ps.print_stats(self.fraction)

        self.stream.write("\nprofile: {}: exit\n".format(self.contextstr))
        # save #
        ps.dump_stats(filename='profiling.prof')
		# gprof2dot -f pstats profiling.prof | dot -Tpng -o output.png && eog output.png

        return False

    def get_profile_data(self):

        value = self.stream.getvalue()
        if self.logger is not None:
            self.logger.info("%s", value)

        return value

    def save_profile_data(self):

        value = self.stream.getvalue()
        if self.logger is not None:
            self.logger.info("%s", value)

        return value


dataset = Set()
dataset.load_recursive( filename='/home/akaris/Documents/code/VASP/v4.7/files/dataset/Bimetalic/dataset_BiMetals_M1TPyPM2Au_catalysis_site2.pkl', verbosity=True)

dict_data = {	'Sc':0 ,'Ti':1 ,'V':2 ,'Cr':3 ,'Mn':4 ,
				'Fe':5 ,'Co':6 ,'Ni':7 ,'Cu':8 ,'Zn':9 ,}

data1 = np.zeros((10, 10))
data2 = np.zeros((10, 10))

for key_data, data in dataset.set.items():
	for key_sys, sys in data.system.items():
		if key_data[-2:] == 'Zn' :#and key_sys[-4:] == '_*OH':
			print( key_data[-2:], key_sys, )
			try:
				print(sys.OUTCAR.total_force.shape)
				print(sys.OUTCAR.POTCAR, sys.OUTCAR.total_force[-1,0,2]- sys.OUTCAR.total_force[-1,2,2] ) #[-1][:3, 0])
			except: pass

			'''
			M1 = -1
			for key, dat in dict_data.items():
				if key in key_data[:5]: M1 = dat; break

			M2 = -1
			for key, dat in dict_data.items():
				if key in key_data[-5:]: M2 = dat; break
			print( key_data, key_sys)
			print(M1, M2)
			
			try:
				d = np.linalg.norm(sys.OUTCAR.total_force[-1, 2,:3]-sys.OUTCAR.total_force[-1, 0,:3])
				print( d )
				data1[M1][M2] = sys.OUTCAR.magnetization[-1, 0,3]
				data2[M1][M2] = d #sys.OUTCAR.magnetization[-1, 1,3]

			except: pass
			'''
plt.plot(data2.T, '-o')
plt.show()

'''
dataset = Set()
dataset.load_recursive( filename='/home/akaris/Documents/code/VASP/v4.7/files/dataset/Bimetalic/dataset_BiMetals_M1TPyPM2Au_catalysis_site1.pkl', verbosity=True)

data_h1 = []
data_h2 = []
for key_data, data in dataset.set.items():
	for key_sys, sys in data.system.items():
		if key_sys[-1] == '*':
			if not sys.OUTCAR is None:

				#print(sys.OUTCAR.total_force[-1,0,2]-4.83312)
				print(sys.OUTCAR.total_force[-1,0,2]-4.83312)

				data_h1.append(sys.OUTCAR.total_force[-1,0,2]-4.83312)
				data_h2.append(sys.OUTCAR.total_force[-1,1,2]-4.83312)
			else:
				data_h1.append(0)
				data_h2.append(0)

				#print(0)
				print(0)

plt.figure(1)
for n in range(10):
	for m in range(10):
		plt.plot(n, data_h2[n*10+m], 'o' )

plt.figure(2)
for m in range(10):
	for n in range(10):
		plt.plot(m, data_h1[n*10+m], 'o' )


plt.show()
'''
#dataset.get_occupancy(save=True, path='/home/akaris/Documents/code/VASP/v4.7/files/dataset/Bimetalic/ocupation_data')
#dataset.load_recursive( filename='/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/dataset_Metales_MTPyP_catalysis_part1.pkl', verbosity=True)
#dataset.get_occupancy(save=True, path='/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/ocupation_data')

	# *******************************************************************************************************************************************************************
	# * === ARGPARSE === ARGPARSE === ARGPARSE === ARGPARSE === ARGPARSE === ARGPARSE === ARGPARSE === ARGPARSE === ARGPARSE === ARGPARSE === ARGPARSE === ARGPARSE === *
	# *******************************************************************************************************************************************************************
def main(argv):
	# === organize arg === #
	inputfile  = argv['input']
	outputfile = argv['output']
	outputfile = inputfile if outputfile is None else outputfile
	task 	   = argv['task']
	files      = argv['files']
	ion    	   = argv['ion']
	orbital    = argv['orbital']
	v 	  	   = True

	# === Make data holder === #
	dataset = Set()

	if task == 'read':
		path = '/'.join(inputfile.split('/')[:-1])
		try:	sys.path.insert(0, f'{path}')
		except: pass
		data_dict = __import__(inputfile.split('/')[-1])
	
		dataset.read_data( dataset=data_dict.Set, files=files, save_format={'pickle'}, 
							output=outputfile, verbosity=v)
		filename = f'{outputfile}.pkl'
		filehandler = open(filename, 'wb')
		pickle.dump(dataset, filehandler)

	if task == 'read_folder':
		path = '/'.join(inputfile.split('/')[:-1])
		try:	sys.path.insert(0, f'{path}')
		except: pass
		
		dataset.read_recursive( path=inputfile, files=files, save_format={'pickle'}, 
							output=outputfile, verbosity=v)
		filename = f'{outputfile}.pkl'
		filehandler = open(filename, 'wb')
		pickle.dump(dataset, filehandler)

	if task == 'xml':
		dataset.load_data( filename=inputfile, verbosity=v)
		dataset.summary(xml=True, path=outputfile)

	if task == 'dat':
		path = '/'.join(inputfile.split('/')[:-1])
		try:	sys.path.insert(0, f'{path}')
		except: pass
		data_dict = __import__(inputfile.split('/')[-1])
	
		dataset.read_data( dataset=data_dict.Set, files=['OSZICAR'], save_format={'pickle', 'light'}, 
						   output=outputfile, verbosity=v, 
						   query={'ORR'		:['overpotencial_OER_4e','overpotencial_ORR_4e', 
													'G1_OER'  ,	'G2_OER'  ,	'G3_OER' ,	'G4_OER' ,
													'G1_ORR'  ,	'G2_ORR'  ,	'G3_ORR' ,	'G4_ORR' ,
													'Eabs_OOH',	'Eabs_O'  ,	'Eabs_OH',
													'Gabs_OH' ,	'Gabs_OOH',	'Gabs_O'	], 
									'E'			:['total'],	
									'PDOS'		:['z']	
								} )
	
	if task == 'XYN':
		path = '/'.join(inputfile.split('/')[:-1])
		file = inputfile.split('/')[-1]
		dataset.load_data(inputfile)
		nnx, nny, name = dataset.get_dataset_PDOS2OP()

		np.savetxt(f'{path}/{file}_X.dat', nnx)
		np.savetxt(f'{path}/{file}_Y.dat', nny)
		np.savetxt(f'{path}/{file}_N.dat', name, fmt='%s')

	if task == 'ocupation':
		path = '/'.join(inputfile.split('/')[:-1])
		file = inputfile.split('/')[-1]
		dataset.load_data( filename=inputfile, verbosity=v)
		dataset.get_occupancy(ion=ion, orbital=orbital, path=f'{path}/ocupation', save=True)
		outputfile = f'{path}/ocupation'

	if task == 'dband' or task == 'band_center':
		path = '/'.join(inputfile.split('/')[:-1])
		file = inputfile.split('/')[-1]
		dataset.load_data( filename=inputfile, verbosity=v)
		dataset.get_center_of_band(ion=ion, orbital=orbital, path=f'{path}/dband', save=True)
		outputfile = f'{path}/dband'


	print(f'Input  >> {inputfile} ')
	print(f'OUTPUT >> {outputfile}')
	#python -m cProfile -o program.prof my_program.py
	# gprof2dot --colour-nodes-by-selftime -f pstats output.pstats |     dot -Tpng -o output.png


'''
#with open('/home/akaris/Documents/code/VASP/v4.7/files/dataset/PC/dataset_CoFePC.pkl', 'rb') as f:
with open('/home/akaris/Documents/code/VASP/v4.7/files/dataset/substitution/substitution.pkl', 'rb') as f:
    data = pickle.load(f)


for key, item in data.set.items():
	try:
		print( item.system)
		#print(key, item.system['PDOS'].DOSCAR.plot(ion=[0], orbital='d') )
		#plt.show()
		#if key in ['CoPC_gamma_5BE_D3_U0','CoPC_gamma_0BE_D3_0U','FePC_gamma_5BE_D3','FePC_gamma_0BE_D3']:

		print(key, item.system['PDOS'].DOSCAR.get_occupancy(ion=[0], orbital=['d_z2-r2']) )
		#print(key, item.analysis_results['ORR'].ORR)
	except: pass
asd
'''

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument('-t','--task', help="task to accomplish \n   read  :  Read files from dataset  \n    summarise  :  resume data from dataset ",
	                    type=str, default='read', required=True)
	
	parser.add_argument('-i','--input', help="path to inputfile",
	                    type=str, default='', required=True)

	parser.add_argument('-o','--output', help="name of data output file",
	                    type=str, default=None, required=False)

	parser.add_argument('-f','--files', help="File list",
	                    type=str, default='all', nargs='+', required=False)

	parser.add_argument('-ion','--ion', help="list of ions",
	                    type=int, default=None, nargs='+', required=False)

	parser.add_argument('-orbital','--orbital', help="list of orbitales",
	                    type=str, default='all', nargs='+', required=False)

	args = vars(parser.parse_args())
	main(args)

