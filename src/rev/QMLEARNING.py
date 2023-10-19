#import yahoo_fin.stock_info as yf
#import requests, ftplib, io, re, json, datetime, time, Logspp
import pandas as pd
import random, Logs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

try: import pandas as pn
except: print('ERROR :: SIMULATION.recursive_read_pandas() :: Can NOT import pandas. \n install with : "pip3 install pandas" ') 

# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #

class QMlearning(object):
	def __init__(self, name:str=None,
						X:np.ndarray=None, Y:np.ndarray=None, Z:np.ndarray=None, occupancy:np.ndarray=None,
						N:np.ndarray=None, Ysplit:int=500):
		self._name = name

		self._X = X
		self._Y = Y
		self._Z = Z
		self.occupancy = occupancy
		self._N = N
		self.Ysplit = Ysplit

		self.df = None

		# --- NN --- #
		self.act_func = 'tanh'
		self.neural_regresor_model = None

		self.cnames = {
'aliceblue':            '#F0F8FF','antiquewhite':         '#FAEBD7','aqua':                 '#00FFFF','aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF','beige':                '#F5F5DC','bisque':               '#FFE4C4','black':                '#000000',
'blanchedalmond':       '#FFEBCD','blue':                 '#0000FF','blueviolet':           '#8A2BE2','brown':                '#A52A2A',
'burlywood':            '#DEB887','cadetblue':            '#5F9EA0','chartreuse':           '#7FFF00','chocolate':            '#D2691E',
'coral':                '#FF7F50','cornflowerblue':       '#6495ED','cornsilk':             '#FFF8DC','crimson':              '#DC143C',
'cyan':                 '#00FFFF','darkblue':             '#00008B','darkcyan':             '#008B8B','darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9','darkgreen':            '#006400','darkkhaki':            '#BDB76B','darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F','darkorange':           '#FF8C00','darkorchid':           '#9932CC','darkred':              '#8B0000',
'darksalmon':           '#E9967A','darkseagreen':         '#8FBC8F','darkslateblue':        '#483D8B','darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1','darkviolet':           '#9400D3','deeppink':             '#FF1493','deepskyblue':          '#00BFFF',
'dimgray':              '#696969','dodgerblue':           '#1E90FF','firebrick':            '#B22222','floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22','fuchsia':              '#FF00FF','gainsboro':            '#DCDCDC','ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700','goldenrod':            '#DAA520','gray':                 '#808080','green':                '#008000',
'greenyellow':          '#ADFF2F','honeydew':             '#F0FFF0','hotpink':              '#FF69B4','indianred':            '#CD5C5C',
'indigo':               '#4B0082','ivory':                '#FFFFF0','khaki':                '#F0E68C','lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5','lawngreen':            '#7CFC00','lemonchiffon':         '#FFFACD','lightblue':            '#ADD8E6',
'lightcoral':           '#F08080','lightcyan':            '#E0FFFF','lightgoldenrodyellow': '#FAFAD2','lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3','lightpink':            '#FFB6C1','lightsalmon':          '#FFA07A','lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA','lightslategray':       '#778899','lightsteelblue':       '#B0C4DE','lightyellow':          '#FFFFE0',
'lime':                 '#00FF00','limegreen':            '#32CD32','linen':                '#FAF0E6','magenta':              '#FF00FF',
'maroon':               '#800000','mediumaquamarine':     '#66CDAA','mediumblue':           '#0000CD','mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB','mediumseagreen':       '#3CB371','mediumslateblue':      '#7B68EE','mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC','mediumvioletred':      '#C71585','midnightblue':         '#191970','mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1','moccasin':             '#FFE4B5','navajowhite':          '#FFDEAD','navy':                 '#000080',
'oldlace':              '#FDF5E6','olive':                '#808000','olivedrab':            '#6B8E23','orange':               '#FFA500',
'orangered':            '#FF4500','orchid':               '#DA70D6','palegoldenrod':        '#EEE8AA','palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE','palevioletred':        '#DB7093','papayawhip':           '#FFEFD5','peachpuff':            '#FFDAB9',
'peru':                 '#CD853F','pink':                 '#FFC0CB','plum':                 '#DDA0DD','powderblue':           '#B0E0E6',
'purple':               '#800080','red':                  '#FF0000','rosybrown':            '#BC8F8F','royalblue':            '#4169E1',
'saddlebrown':          '#8B4513','salmon':               '#FA8072','sandybrown':           '#FAA460','seagreen':             '#2E8B57',
'seashell':             '#FFF5EE','sienna':               '#A0522D','silver':               '#C0C0C0','skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD','slategray':            '#708090','snow':                 '#FFFAFA','springgreen':          '#00FF7F',
'steelblue':            '#4682B4','tan':                  '#D2B48C','teal':                 '#008080','thistle':              '#D8BFD8',
'tomato':               '#FF6347','turquoise':            '#40E0D0','violet':               '#EE82EE','wheat':                '#F5DEB3',
'white':                '#FFFFFF','whitesmoke':           '#F5F5F5','yellow':               '#FFFF00','yellowgreen':          '#9ACD32'}

		self.functional_metadata = {
			'D3_' 		: {'marker'	: 'o', 'color':(1,0,0)},
			'D3' 		: {'marker'	: 'o', 'color':(1,0,0)},
			'BJ' 		: {'marker'	: 'v', 'color':(0.5,0,0)},
			'D3BJ' 		: {'marker'	: 'v', 'color':(0.5,0,0)},

			'DF_' 		: {'marker'	: '1', 'color':(0,1,0)},
			'DF' 		: {'marker'	: '1', 'color':(0,1,0)},

			'DF2_' 		: {'marker'	: 's', 'color':(0.2,0.7,0)},
			'DF2' 		: {'marker'	: 's', 'color':(0.2,0.7,0)},

			'DF2B86' 	: {'marker'	: '+', 'color':(0.5,0.5,0)},
			'OPT86' 	: {'marker'	: 'P', 'color':(0,0,1)},
			'OPTPBE' 	: {'marker'	: '*', 'color':(0,0,0.5)},
		}

		self.atomic_metadata = {
		    # Name [period, column,                  valence]
		    'Al' : { 'period'	:3, 	'column':13,	'valence':[3],	 			 'number':0,	'color':'aliceblue',	 	 'marker':'o'},
		    'Bi' : { 'period'	:6, 	'column':15,	'valence':[3,5],			 'number':1,	'color':'azure',			 'marker':'o'},
		    'Ca' : { 'period'	:4, 	'column':2,		'valence':[2],	 			 'number':2,	'color':'blanchedalmond',	 'marker':'o'},
		    'Cd' : { 'period'	:5, 	'column':12,	'valence':[2],				 'number':3,	'color':'coral',			 'marker':'v'},
		    'Co' : { 'period'	:4, 	'column':9,		'valence':[2,3],			 'number':4,	'color':'cyan',	 			 'marker':'1'},
		    'Cr' : { 'period'	:4, 	'column':6,		'valence':[2,3,6],			 'number':5,	'color':'darkgray',	 		 'marker':'s'},
		    'Cu' : { 'period'	:4, 	'column':11,	'valence':[1,2],	 		 'number':6,	'color':'darkolivegreen',	 'marker':'p'},
		    'Ir' : { 'period'	:6, 	'column':9,		'valence':[1,2],			 'number':7,	'color':'darkturquoise',	 'marker':'o'},
		    'Fe' : { 'period'	:4, 	'column':8,		'valence':[2,3],			 'number':8,	'color':'dimgray',	 		 'marker':'x'},
		    'Mg' : { 'period'	:3, 	'column':2,		'valence':[2],				 'number':9,	'color':'indigo',	 		 'marker':'o'},
		    'Mn' : { 'period'	:4, 	'column':7,		'valence':[2,3,4,6,7],		 'number':10,	'color':'lavenderblush',	 'marker':'<'},
		    'Mo' : { 'period'	:5, 	'column':6,		'valence':[2,3,4,5,6],		 'number':11,	'color':'lightskyblue',	 	 'marker':'o'},
		    'Ni' : { 'period'	:4, 	'column':10,	'valence':[2,3],	 		 'number':12,	'color':'maroon',	 		 'marker':'>'},
		    'Pb' : { 'period'	:6, 	'column':14,	'valence':[2,4],			 'number':13,	'color':'mediumturquoise',	 'marker':'o'},
		    'Pd' : { 'period'	:5, 	'column':10,	'valence':[2,4],			 'number':14,	'color':'skyblue',	 		 'marker':'o'},
		    'Pt' : { 'period'	:6, 	'column':10,	'valence':[2,4],		 	 'number':15,	'color':'orangered',		 'marker':'2'},
		    'Ru' : { 'period'	:5, 	'column':8,		'valence':[2,3,4,5,6,7,8],	 'number':16,	'color':'paleturquoise',	 'marker':'o'},
		    'Rh' : { 'period'	:5, 	'column':9,		'valence':[2,3,4,5,6],		 'number':17,	'color':'purple',	 		 'marker':'o'},
		    'Sc' : { 'period'	:4, 	'column':3,		'valence':[3],				 'number':18,	'color':'saddlebrown',	 	 'marker':'P'},
		    'Ti' : { 'period'	:4, 	'column':4,		'valence':[2,3,4],	 		 'number':19,	'color':'plum',	 			 'marker':'*'},
		    'Tc' : { 'period'	:5, 	'column':7,		'valence':[2,3,4],	 		 'number':20,	'color':'tomato',	 		 'marker':'o'},
		     'V' : { 'period'	:4, 	'column':5,		'valence':[2,3,4,5],		 'number':21,	'color':'teal',	 			 'marker':'h'},
		    'Zn' : { 'period'	:4, 	'column':12,	'valence':[2,3,4,5],		 'number':22,	'color':'slategray',		 'marker':'D'},
		   } # for key, item in self.atomic_metadata.items():	item.append( random.choice(list(self.cnames)) ) 

		self.atomic_metadata_list = {
			'Al' : [    3,      13,      [3]              	, 1     ,	'azure'			    ],
		    'Bi' : [    6,      15,      [3,5]              , 1     ,	'azure'				],
		    'Ca' : [    4,       2,      [2]                , 2     ,	'blanchedalmond'	],
		    'Cd' : [    5,      12,      [2]                , 3     ,	'coral'				],
		    'Co' : [    4,       9,      [2,3]              , 4     ,	'cyan'				],
		    'Cr' : [    4,       6,      [2,3,6]            , 5     ,	'darkgray'			],
		    'Cu' : [    4,      11,      [1,2]              , 6     ,	'darkolivegreen'	],
		    'Ir' : [    6,       9,      [1,2]              , 7     ,	'darkturquoise'		],
		    'Fe' : [    4,       8,      [2,3]              , 8     ,	'dimgray'			],
		    'Mg' : [    3,       2,      [2]                , 9     ,	'indigo'			],
		    'Mn' : [    4,       7,      [2,3,4,6,7]        , 10    ,	'lavenderblush'		],
		    'Mo' : [    5,       6,      [2,3,4,5,6]        , 11    ,	'lightskyblue'		],
		    'Ni' : [    4,      10,      [2,3]              , 12    ,	'maroon'			],
		    'Pb' : [    6,      14,      [2,4]              , 13    ,	'mediumturquoise'	],
		    'Pd' : [    5,      10,      [2,4]              , 14    ,	'skyblue'			],
		    'Pt' : [    6,      10,      [2,4]              , 15    ,	'orangered'			],
		    'Ru' : [    5,       8,      [2,3,4,5,6,7,8]    , 16    ,	'paleturquoise'		],
		    'Rh' : [    5,       9,      [2,3,4,5,6]        , 17    ,	'purple'			],
		    'Sc' : [    4,       3,      [3]                , 18    ,	'saddlebrown'		],
		    'Ti' : [    4,       4,      [2,3,4]            , 19    ,	'plum'				],
		    'Tc' : [    5,       7,      [2,3,4]            , 20    ,	'tomato'			],
		     'V' : [    4,       5,      [2,3,4,5]          , 21    ,	'teal'				],
		     'h' : [    4,       5,      [2,3,4,5]          , 21    ,	'teal'				],
		    'Zn' : [    4,      12,      [2,3,4,5]          , 22    ,	'slategray'			]
		} 
		for key, item in self.atomic_metadata_list.items():	item.append( random.choice(list(self.cnames)) )

	# ------------------- name ------------------- #
	@property
	def name(self, ) -> str:
		if self._name is None:	return 'Unknow'	
		else:					return self._name

	@name.setter
	def name(self, name:str):	self._name = name

	@name.deleter
	def name(self, ):	del self._name

	# ------------------- X ------------------- #
	@property
	def X(self, ) -> np.ndarray:
		if self._X is None:	return 'Unknow'	
		else:					return self._X

	@X.setter
	def name(self, X:np.ndarray):	self._X = X

	@X.deleter
	def X(self, ):	del self._X

	# ------------------- X ------------------- #
	@property
	def Y(self, ) -> np.ndarray:
		if self._Y is None:	return 'Unknow'	
		else:					return self._Y

	@Y.setter
	def name(self, Y:np.ndarray):	self._Y = Y

	@Y.deleter
	def Y(self, ):	del self._Y


	# ------------------- X ------------------- #
	@property
	def Z(self, ) -> np.ndarray:
		if self._Z is None:	return 'Unknow'	
		else:					return self._Z

	@Z.setter
	def name(self, Z:np.ndarray):	self._Z = Z

	@Z.deleter
	def Z(self, ):	del self._Z


	# ------------------- N ------------------- #
	@property
	def N(self, ) -> np.ndarray:
		if self._N is None:	return 'Unknow'	
		else:					return self._N

	@N.setter
	def name(self, N:np.ndarray):	self._N = N

	@N.deleter
	def N(self, ):	del self._N

	def load(self, file:str=None) -> np.ndarray:
		if file[-3:] == '.npy':	return np.load(file)
		else:					
			try:	return np.loadtxt(file)
			except:	return np.loadtxt(file, dtype=str)

	@Logs.LogDecorator()
	def load_folder(self, path:str=None, 
						save:bool=True, v:bool=True) -> list:

		try:
			from os import listdir
			from os.path import isfile, join
		except:
			print('>> (!) ERROR :: can not import os :: try pip3 install os')

		loaded_dict = {}
		for n in [f for f in listdir(path) if isfile(join(path, f))]:

			if n[-5:] == 'X.dat':
				if v: print( f' >> Loading X.dat :: from {path}/{n}')
				loaded_dict['X'] = self.load(file=f'{path}/{n}')
				if v: print( f' (*) Loaded X.dat :: {loaded_dict["X"].shape}')
				if save: 
					if self._X is None: self._X = loaded_dict['X']
					else:				self._X = np.concatenate( (self._X, loaded_dict['X']), axis=0 )

			if n[-5:] == 'Y.dat':
				if v: print( f' >> Loading Y.dat :: from {path}/{n}')
				loaded_dict['Y'] = self.load(file=f'{path}/{n}')
				if v: print( f' (*) Loaded Y.dat :: {loaded_dict["Y"].shape}')
				if save: 
					if self._Y is None: self._Y = loaded_dict['Y']
					else:				self._Y = np.concatenate( (self._Y, loaded_dict['Y']), axis=0 )
			if n[-5:] == 'Z.dat':
				if v: print( f' >> Loading Z.dat :: from {path}/{n}')
				loaded_dict['Z'] = self.load(file=f'{path}/{n}')
				if v: print( f' (*) Loaded Z.dat :: {loaded_dict["Z"].shape}')
				if save: 
					if self._Z is None: self._Z = loaded_dict['Z']
					else:				self._Z = np.concatenate( (self._Z, loaded_dict['Z']), axis=0 )

			if n[-5:] == 'N.dat':
				if v: print( f' >> Loading N.dat :: from {path}/{n}')
				loaded_dict['N'] = self.load(file=f'{path}/{n}')
				if v: print( f' (*) Loaded N.dat :: {loaded_dict["N"].shape}')
				if save: 
					if self._N is None: self._N = loaded_dict['N']
					else:				self._N = np.concatenate( (self._N, loaded_dict['N']) )
		return loaded_dict

	def make_dataframe(self, X:np.ndarray=None, Xlabel:list=None, N:np.ndarray=None,
						save:bool=True):
		df = pd.DataFrame(X, columns=Xlabel) if not Xlabel is None else pd.DataFrame(X)
		if not N is None: df.index = N
		if save: self.df = df

		return df

	def dataframe_add(self, X:np.ndarray=None, Xlabel:list=None, N:np.ndarray=None,
						 df=None, save:bool=True):
		df = df if not df is None else self.df
		df2 = pd.DataFrame(X, columns=Xlabel) if not Xlabel is None else pd.DataFrame(X)
		if not N is None: df2.index = N
		df = pd.concat([df, df2], axis=1)
		if save: self.df = df

		return df

	def filter(self, filters:list=None, X:np.ndarray=None, Y:np.ndarray=None, Z:np.ndarray=None, N:np.ndarray=None,
					v:bool=True, save:bool=True):
		X = X if not X is None else self.X
		Y = Y if not Y is None else self.Y
		Z = Z if not Z is None else self.Z
		N = N if not N is None else self.N

		for f in filters:
			if f['type'] == 'N':
				if f['condition'] == 'whitelist':
					filter_array = np.array([ np.any( np.array([ m in n for m in f['list']],dtype=bool) ) == True for n in N])
			
			if f['type'] == 'X':
				if f['condition'] == 'isNAN':
					filter_array = ~np.isnan(X).any(axis=1) 

				if f['condition'] == 'minor':
					filter_array = (X<f['value']).all(axis=1)

			if f['type'] == 'Y':
				if f['condition'] == 'isNAN':
					filter_array = ~np.isnan(Y).any(axis=1) 


		if save:
			try:	self._X = X[filter_array]
			except: pass

			try:	self._Y = Y[filter_array]
			except: pass

			try:	self._Z = Z[filter_array]
			except: pass

			try:	self._N = N[filter_array]
			except: pass

		return filter_array

	def ORR_analysis(self, X:np.ndarray=None, Y:np.ndarray=None, Z:np.ndarray=None, N:np.ndarray=None, occupancy:np.ndarray=None,
						curve_fit:bool=True, marker:list=None, color:list=None, Ylabel:list=None, Xlabel:list=None,
						degree:int=1, save:bool=True, v:bool=True, text:bool=False):
		X = X if not X is None else self.X # ORR
		#Y = Y if not Y is None else self.Y # PDOS + CHG/MAG  
		Z = Z if not Z is None else self.Z # metadata
		N = N if not N is None else self.N # system names

		color   = color   if not color   is None else [(0.8,0.3,0.3)]*int(X.shape[0])
		marker  = marker  if not marker  is None else ['*']*int(X.shape[0])
		Xlabel  = Xlabel  if not Xlabel  is None else [ 	'overpotencial_ORR_4e', 
													'Gabs_OOH', 'Eabs_OOH', 
													'Gabs_OH',  'Eabs_OH', 
													'Gabs_O', 'Eabs_O', 
													'G1_ORR', 'G2_ORR', 'G3_ORR', 'G4_ORR', 
													'limiting_step_ORR'] # representacion de las variables X

		samples, Ydim = Y.shape if not Y is None else 0, 0
		samples, Xdim = X.shape

		if not X is None:
			# ======== 3x3 PLOTs ======== ======== 3x3 PLOTs ======== ======== 3x3 PLOTs ========
			fig, ax = plt.subplots(3, 3, figsize=(10, 10) )
			([ax11, ax12, ax13], [ax21, ax22, ax23], [ax31, ax32, ax33]) = ax
			fig.tight_layout()
			str_keyX = [['Gabs_OOH', 'Gabs_OH', 'Gabs_O'], ['G1_ORR', 'G2_ORR', 'G3_ORR'], ['Gabs_OH', 'Gabs_OOH', 'Gabs_OOH']]
			str_keyY = [['overpotencial_ORR_4e', 'overpotencial_ORR_4e', 'overpotencial_ORR_4e'], ['overpotencial_ORR_4e', 'overpotencial_ORR_4e', 'overpotencial_ORR_4e'], ['Gabs_O', 'Gabs_OH', 'Gabs_O']]
			
			# >> set AX label
			for i1 in range(9):  ax[int(i1/3)][int(i1%3)].set(	xlabel=f'{str_keyX[int(i1/3)][int(i1%3)]} (eV)', 
																ylabel=f'{str_keyY[int(i1/3)][int(i1%3)]}  (eV)') 

			for n in range(samples):
				# >> plot 3x3
				for i1 in range(3):
					for i2 in range(3):
						ax[i1][i2].plot( X[n,Xlabel.index(str_keyX[i1][i2])], X[n,Xlabel.index(str_keyY[i1][i2])], marker=marker[n] , color=color[n] )
						if text: ax[i1][i2].text( X[n,Xlabel.index(str_keyX[i1][i2])], X[n,Xlabel.index(str_keyY[i1][i2])], N[n], color=color[n] )

			if curve_fit:
				from scipy.optimize import curve_fit

				def modelo(x, a, b, c, d):
					Y = np.zeros_like(x)
					Y[x<=c] = a*x[x<=c] + b 
					Y[x> c] = d*x[x> c] + (a-d)*c+b
					return Y

				# >> plot curve_fit
				for i1 in range(3):
					for i2 in range(3):
						try:
							# >> find initial values for fitting 
							Xmin, Xmax = np.min(X[:,Xlabel.index(str_keyX[i1][i2])])*0.9, np.max(X[:,Xlabel.index(str_keyX[i1][i2])])*1.1
							Xnew = np.linspace(Xmin, Xmax, 100)
							
							# >> curve fitting 
							popt, pcov = curve_fit(modelo, X[:,Xlabel.index(str_keyX[i1][i2])], X[:,Xlabel.index(str_keyY[i1][i2])], p0=[1, 5, X[np.argmin(X[:,Xlabel.index(str_keyY[i1][i2])]),Xlabel.index(str_keyX[i1][i2])], -1])
							
							# >> calculate error
							ss_res = np.sum( (X[:,Xlabel.index(str_keyY[i1][i2])] - modelo(X[:,Xlabel.index(str_keyX[i1][i2])], *popt))**2  )	# Suma de los cuadrados de los residuos
							ss_tot = np.sum( (X[:,Xlabel.index(str_keyY[i1][i2])] - np.mean(X[:,Xlabel.index(str_keyY[i1][i2])]) )**2  ) 		# Suma total de cuadrados

							# >> plot fitting 
							ax[i1][i2].plot(Xnew, modelo(Xnew, *popt), marker=None, linestyle='dashed', color=(0.9,0.3,0.3), label=f'{popt[0]:.2}{str_keyX[i1][i2]}+{popt[1]:.2} \n {popt[3]:.2}{str_keyX[i1][i2]}+{(popt[0]-popt[3])*popt[2]:.2} \n cut at {popt[2]:.2} | R2={1-(ss_res/ss_tot):.2}')
							ax[i1][i2].legend(loc='best')
						except: pass

		if not Y is None and not X is None:
			# ======== 4x4 PLOTs ======== ======== 4x4 PLOTs ======== ======== 4x4 PLOTs ========
			for pp in range(4):
				fig, ax = plt.subplots(3, 4, figsize=(13, 10) )
				([ax11, ax12, ax13, ax14], [ax21, ax22, ax23, ax24], [ax31, ax32, ax33, ax34]) = ax
				str_keyX = ['overpotencial_ORR_4e', 'Gabs_OOH', 'Gabs_OH', 'Gabs_O']
				for i1 in range(4):
					for i2 in range(3):
						ax[i2][i1].set(xlabel=f'BC_{Ylabel[pp*3+i2]}', ylabel=f'{str_keyX[i1]} (eV)')

				fig.tight_layout()

				for n in range(samples):
					# >> PLOT data 
					for i1 in range(4):
						for i2 in range(3):
							if i1 == 0:
								print(Y.shape, pp*3+i2, Y[n,pp*3+i2], color[n], N[n],  X[n,Xlabel.index(str_keyX[i1])])

							ax[i2][i1].plot( Y[n,pp*3+i2], X[n,Xlabel.index(str_keyX[i1])], marker=marker[n] , color=color[n] )
							if text: ax[i2][i1].text( Y[n,pp*3+i2], X[n,Xlabel.index(str_keyX[i1])], N[n], color=color[n] )

				if curve_fit:
					# >> fitting curve
					from scipy.optimize import curve_fit

					# >> fitting model 
					def modelo(x, a, b, c, d):
						Y = np.zeros_like(x)
						Y[x<=c] = a*x[x<=c] + b 
						Y[x> c] = d*x[x> c] + (a-d)*c+b
						return Y

					for i1 in range(4):
						for i2 in range(3):
							try: 
								# >> plot fitting
								Xmin, Xmax = np.min(Y[:,pp*3+i2][~np.isnan(Y[:,pp*3+i2])])*0.9, np.max(Y[:,pp*3+i2][~np.isnan(Y[:,pp*3+i2])])*1.1
								Xnew = np.linspace(Xmin, Xmax, 100)
								popt, pcov = curve_fit(modelo, Y[:,pp*3+i2][~np.isnan(Y[:,pp*3+i2])], X[:,Xlabel.index(str_keyX[i1])][~np.isnan(Y[:,pp*3+i2])], p0=[-1, 5, Y[np.argmin(X[:,Xlabel.index(str_keyX[i1])]),pp*3+i2], 1])
								ss_res = np.sum( (X[:,Xlabel.index(str_keyX[i1])] - modelo(Y[:,pp*3+i2], *popt))**2  )	# Suma de los cuadrados de los residuos
								ss_tot = np.sum( (X[:,Xlabel.index(str_keyX[i1])] - np.mean(X[:,Xlabel.index(str_keyX[i1])]) )**2  ) 		# Suma total de cuadrados
								ax[i2][i1].plot(Xnew, modelo(Xnew, *popt), marker=None, linestyle='dashed', color=(0.9,0.3,0.3), label=f'{popt[0]:.2}{LCnames[pp*3+i2]}+{popt[1]:.2} \n {popt[3]:.2}{LCnames[pp*3+i2]}+{(popt[0]-popt[3])*popt[2]:.2} \n cut at {popt[2]:.2} | R2={1-(ss_res/ss_tot):.2}')
								ax[i2][i1].legend(loc='best')
							except: pass 

		if not occupancy is None and not X is None:
			# ======== 4x4 PLOTs ======== ======== 4x4 PLOTs ======== ======== 4x4 PLOTs ========
			for pp in range(2):
				fig, ax = plt.subplots(5, 4, figsize=(13, 10) )
				str_keyX = ['overpotencial_ORR_4e', 'Gabs_OOH', 'Gabs_OH', 'Gabs_O']

				fig.tight_layout()
				for n in range(samples):
					for sys in occupancy[:,0]:
						if self.N[n] in sys:
							# >> PLOT data 
							for i1 in range(4):
								for i2 in range(5):
									ax[i2][i1].plot( float(occupancy[n,pp*5+i2+1]), X[n,Xlabel.index(str_keyX[i1])], marker=marker[n] , color=color[n] )
									if text: ax[i2][i1].text( Y[n,pp*3+i2], X[n,Xlabel.index(str_keyX[i1])], N[n], color=color[n] )

				'''
				if curve_fit:
					# >> fitting curve
					from scipy.optimize import curve_fit

					# >> fitting model 
					def modelo(x, a, b, c, d):
						Y = np.zeros_like(x)
						Y[x<=c] = a*x[x<=c] + b 
						Y[x> c] = d*x[x> c] + (a-d)*c+b
						return Y

					for i1 in range(4):
						for i2 in range(3):
							try: 
								# >> plot fitting
								Xmin, Xmax = np.min(Y[:,pp*3+i2][~np.isnan(Y[:,pp*3+i2])])*0.9, np.max(Y[:,pp*3+i2][~np.isnan(Y[:,pp*3+i2])])*1.1
								Xnew = np.linspace(Xmin, Xmax, 100)
								popt, pcov = curve_fit(modelo, Y[:,pp*3+i2][~np.isnan(Y[:,pp*3+i2])], X[:,Xlabel.index(str_keyX[i1])][~np.isnan(Y[:,pp*3+i2])], p0=[-1, 5, Y[np.argmin(X[:,Xlabel.index(str_keyX[i1])]),pp*3+i2], 1])
								ss_res = np.sum( (X[:,Xlabel.index(str_keyX[i1])] - modelo(Y[:,pp*3+i2], *popt))**2  )	# Suma de los cuadrados de los residuos
								ss_tot = np.sum( (X[:,Xlabel.index(str_keyX[i1])] - np.mean(X[:,Xlabel.index(str_keyX[i1])]) )**2  ) 		# Suma total de cuadrados
								ax[i2][i1].plot(Xnew, modelo(Xnew, *popt), marker=None, linestyle='dashed', color=(0.9,0.3,0.3), label=f'{popt[0]:.2}{LCnames[pp*3+i2]}+{popt[1]:.2} \n {popt[3]:.2}{LCnames[pp*3+i2]}+{(popt[0]-popt[3])*popt[2]:.2} \n cut at {popt[2]:.2} | R2={1-(ss_res/ss_tot):.2}')
								ax[i2][i1].legend(loc='best')
							except: pass 
				'''

		plt.show()

	def deeplearnning_regresor_fit(self, X:np.ndarray=None, Y:np.ndarray=None, Z:np.ndarray=None, N:np.ndarray=None, 
								act_func:str='tanh', epochs:int=100, 
								marker:list=None, color:list=None, Ylabel:list=None, Xlabel:list=None,
								v:bool=True, save:bool=True):
		
		X 			= X 		if not X is None 		else self.X
		Y 			= Y 		if not Y is None 		else self.Y
		Z 			= Z 		if not Z is None 		else self.Z
		N 			= N 		if not N is None 		else self.N
		act_func 	= act_func 	if not act_func is None else self.act_func

		color   = color   if not color   is None else [(0.8,0.3,0.3)]*int(X.shape[0])
		marker  = marker  if not marker  is None else ['*']*int(X.shape[0])
		Xlabel  = Xlabel  if not Xlabel  is None else [ 	'overpotencial_ORR_4e', 
													'Gabs_OOH', 'Eabs_OOH', 
													'Gabs_OH',  'Eabs_OH', 
													'Gabs_O', 'Eabs_O', 
													'G1_ORR', 'G2_ORR', 'G3_ORR', 'G4_ORR', 
													'limiting_step_ORR'] # representacion de las variables X

		# ==== DEFINE models ==== #
		def get_model_l0(width:int ,out:int):
			my_l2 = l2(0.03)
			model = Sequential()
			model.add(Reshape((width, 1), input_shape=(width,)))
			model.add(Conv1D(filters=40, kernel_size=1000, activation='relu', kernel_regularizer=my_l2))
			model.add(MaxPooling1D(400))
			model.add(Flatten())
			model.add(Dense(40, activation='relu', kernel_regularizer=my_l2))
			model.add(Dense(40, activation='relu', kernel_regularizer=my_l2))
			model.add(Dense(60, activation='relu', kernel_regularizer=my_l2))
			model.add(Dense(1, activation='linear', kernel_regularizer=my_l2))
			model.compile(loss='mean_squared_error', optimizer='adam', metrics=[coeff_determination, ])

			return model

		def get_model_l1(width:int ,out:int):
			my_l2 = l2(0.004)
			model = Sequential()
			model.add(Reshape((width, 1), input_shape=(width,)))
			model.add(Conv1D(filters=20, kernel_size=2, activation=act_func, kernel_regularizer=my_l2))
			model.add(MaxPooling1D(5))
			model.add(Flatten())
			model.add(Dense(40, activation='relu', kernel_regularizer=my_l2))
			model.add(Dense(40, activation='relu', kernel_regularizer=my_l2))
			model.add(Dense(1, activation='linear', kernel_regularizer=my_l2))
			model.compile(loss='mean_squared_error', optimizer='adam', metrics=[coeff_determination, ])

			return model

		def get_model_l2(width:int):
			my_l2 = l2(0.04)
			model = Sequential()
			model.add(Reshape((width, 1), input_shape=(width,)))
			model.add(Conv1D(filters=40, kernel_size=50, activation=act_func, kernel_regularizer=my_l2))
			model.add(MaxPooling1D(60))
			model.add(Flatten())
			model.add(Dense(40, activation='relu', kernel_regularizer=my_l2))
			model.add(Dense(1, activation='linear', kernel_regularizer=my_l2))
			model.compile(loss='mean_squared_error', optimizer='adam', metrics=[coeff_determination, ])

			return model

		def coeff_determination(y_true, y_pred):
			SS_res = K.sum(K.square(y_true-y_pred))
			SS_tot = K.sum(K.square(y_true - K.mean(y_true)))

			return 1 - SS_res/SS_tot

		samples = X.shape[0]
		samples = Y.shape[0]

		try:	
			import keras.backend as K
			import keras.callbacks
			from keras.models import Sequential
			from keras.layers import Conv1D, Flatten, BatchNormalization, MaxPooling1D, Dense, Reshape
			from keras.regularizers import l2, l1
			from sklearn.model_selection import train_test_split
			import seaborn as sns
			sns.set_style('ticks')
		except: print('(!) ERROR :: can NOT import lib')

		# === MODEL data === #
		print( '>> == Data shape ==' )
		print( f' (OK) X : {X.shape}' )
		print( f' (OK) Y : {Y.shape}' )
		print( f' (OK) N : {N.shape}' )

		# === set MODEL === #
		model = get_model_l0(X.shape[1], 1 if len(Y.shape)==1 else Y.shape[1])
		model.summary()
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
		history =  model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=2)

		# === PLOT data === #
		# regresion ECM vs epoch #
		fig, ([ax11, ax12, ax13]) = plt.subplots(1, 3, figsize=(15, 5) )
		fig.tight_layout()

		ax11.plot( history.history['loss'] )
		ax11.set(xlabel='epoch'	, ylabel='ECM', title='ECM vs. epochs'	)

		# data vs PREDICT #
		y_regr = model.predict(X_train)
		ax12.plot(y_train, y_regr, 'o', color=self.cnames['tomato'], label='trainnig data')
		y_regr = model.predict(X_test)
		ax12.plot(y_test, y_regr, 'o', color=self.cnames['royalblue'], label='test data')
		ax12.set(xlabel='y data'	, ylabel='y prediction', title='Regresion Curve'	)
		ss_res = np.sum( (np.concatenate( (y_train, y_test) ) - np.concatenate( (model.predict(X_train), model.predict(X_test)))[:,0] )**2  )	# Suma de los cuadrados de los residuos
		ss_tot = np.sum( (np.concatenate( (y_train, y_test) ) - np.mean(np.concatenate( (y_train, y_test) )) )**2  ) 		# Suma total de cuadrados
		ax12.plot( [ 0.0, 5], [   0, 5.0] , ':' , alpha=0.5, c=(0,0,0), label=f'R2 = {1-(ss_res/ss_tot):.2}')
		ax12.plot( [ 0.0, 5], [ 0.5, 5.5] , '-' , alpha=0.5, c=(1,0,0), label='500meV error')
		ax12.plot( [ 0.0, 5], [-0.5, 4.5] , '-' , alpha=0.5, c=(1,0,0))
		ax12.legend(loc='best')

		# === RMSE hist === #
		ax13.hist( y_train-model.predict(X_train),  ) 
		ax13.hist( y_test-model.predict(X_test)  ,  ) 
		ax13.set(xlabel='nominal - prediction'	, ylabel='ECM', title='Residues'	)

		plt.show()

		# === SAVE model === #
		if save: self.neural_regresor_model = model

		return True

	def deeplearnning_predict(self, X:np.ndarray=None, Y:np.ndarray=None, Z:np.ndarray=None, N:np.ndarray=None, Ysplit:int=None, 
								marker:list=None, color:list=None, Ylabel:list=None, Xlabel:list=None,
								v:bool=True):
		
		X 			= X 		if not X is None 		else self.X
		Y 			= Y 		if not Y is None 		else self.Y
		Z 			= Z 		if not Z is None 		else self.Z
		N 			= N 		if not N is None 		else self.N

		def coeff_determination(y_true, y_pred):
			SS_res = K.sum(K.square(y_true-y_pred))
			SS_tot = K.sum(K.square(y_true - K.mean(y_true)))

			return 1 - SS_res/SS_tot

		try:	
			import keras.backend as K
			import keras.callbacks
			from keras.models import Sequential
			from keras.layers import Conv1D, Flatten, BatchNormalization, MaxPooling1D, Dense, Reshape
			from keras.regularizers import l2, l1
			from sklearn.model_selection import train_test_split
			import seaborn as sns
			sns.set_style('ticks')
		except: print('(!) ERROR :: can NOT import lib')

		# === MODEL data === #
		print( '>> == Data shape ==' )
		print( f' (OK) X : {X.shape}' )
		print( f' (OK) Y : {Y.shape}' )
		print( f' (OK) N : {N.shape}' )

		# === load MODEL === #
		model = self.neural_regresor_model

		# === PLOT data === #
		fig, ([ax11, ax12]) = plt.subplots(1, 2, figsize=(10, 5) )
		ax11.set(xlabel='y data'	, ylabel='y prediction', title='Regresion Curve'	)
		y_regr = model.predict(X)
		Xmin, Xmax = np.min(Y), np.max(Y)
		Ymin, Ymax = np.min(y_regr), np.max(y_regr)
		ss_res = np.sum( (Y - y_regr[:,0])**2  )			# Suma de los cuadrados de los residuos
		ss_tot = np.sum( (Y - np.mean(Y))**2  ) 		# Suma total de cuadrados 

		for i, y in enumerate(Y):
			ax11.plot( Y[i], y_regr[i], marker=marker[i] if not marker is None else 'o' , color=color[i]) if not color is None else (1,0,0)

		ax11.plot( [ Xmin, Xmax], [ Xmin, Xmax] , ':' , alpha=0.5, c=(0,0,0), label=f'R2 = {1-(ss_res/ss_tot):.2}')
		ax11.plot( [ Xmin, Xmax], [ Xmin+0.5, Xmax+0.5] , '-' , alpha=0.5, c=(1,0,0)) 
		ax11.plot( [ Xmin, Xmax], [ Xmin-0.5, Xmax-0.5] , '-' , alpha=0.5, c=(1,0,0)) 
		ax11.legend(loc='best')	
		for i, n in enumerate(Y-y_regr[:,0]):
			if abs(n) > 0.5:	ax11.text( Y[i], y_regr[i,0], N[i] , alpha=0.6, c=(0,0,0))

		# RMSE hist #
		ax12.hist( Y-model.predict(X)[:,0] )
		ax12.set(xlabel='nominal - prediction'	, ylabel='ECM', title='Residues'	)

		plt.show()

	def neural_regresor_fit(self, X:np.ndarray=None, Y:np.ndarray=None, Z:np.ndarray=None, N:np.ndarray=None, 
								act_func:str='tanh', epochs:int=1000,
								v:bool=True, save:bool=True):
		
		X 			= X 		if not X is None 		else self.X
		Y 			= Y 		if not Y is None 		else self.Y
		Z 			= Z 		if not Z is None 		else self.Z
		N 			= N 		if not N is None 		else self.N
		act_func 	= act_func 	if not act_func is None else self.act_func

		def get_model_d0(epochs:int=1000):
			return MLPRegressor(hidden_layer_sizes=(250, 250, 250, 250, 250,), activation='relu',
								 alpha=0.0001,random_state=1, max_iter=epochs)


		def coeff_determination(y_true, y_pred):
			SS_res = K.sum(K.square(y_true-y_pred))
			SS_tot = K.sum(K.square(y_true - K.mean(y_true)))

			return 1 - SS_res/SS_tot

		samples= X.shape[0]
		samples = Y.shape[0]

		from sklearn.neural_network import MLPRegressor
		from sklearn.model_selection import train_test_split

		import seaborn as sns
		sns.set_style('ticks')

		# === MODEL data === #
		print( '>> == Data shape ==' )
		print( f' (OK) X : {X.shape}' )
		print( f' (OK) Y : {Y.shape}' )
		print( f' (OK) N : {N.shape}' )

		# === set MODEL === #
		model = get_model_d0(epochs=epochs)
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
		model = model.fit(X_train, y_train)
		model.predict(X_test)

		# === PLOT data === #
		# regresion ECM vs epoch #
		fig, ([ax11, ax12]) = plt.subplots(1, 2, figsize=(15, 5) )
		fig.tight_layout()

		# data vs PREDICT #
		y_regr = model.predict(X_train)
		ax11.plot(y_train, y_regr, 'o', color=self.cnames['tomato'], label='trainnig data')
		y_regr = model.predict(X_test)
		ax11.plot(y_test, y_regr, 'o', color=self.cnames['royalblue'], label='test data')
		ax11.set(xlabel='y data'	, ylabel='y prediction', title='Regresion Curve'	)
		ss_res = np.sum( (np.concatenate( (y_train, y_test) ) - np.concatenate( (model.predict(X_train), model.predict(X_test)) ) )**2  )	# Suma de los cuadrados de los residuos
		ss_tot = np.sum( (np.concatenate( (y_train, y_test) ) - np.mean(np.concatenate( (y_train, y_test) )) )**2  ) 		# Suma total de cuadrados
		ax11.plot( [ 0.0, 5], [   0, 5.0] , ':' , alpha=0.5, c=(0,0,0), label=f'R2 = {1-(ss_res/ss_tot):.2}')
		ax11.plot( [ 0.0, 5], [ 0.5, 5.5] , '-' , alpha=0.5, c=(1,0,0), label='500meV error')
		ax11.plot( [ 0.0, 5], [-0.5, 4.5] , '-' , alpha=0.5, c=(1,0,0))
		ax11.legend(loc='best')

		#  RMSE hist  #
		ax12.hist( y_train-model.predict(X_train),  ) 
		ax12.hist( y_test-model.predict(X_test)  ,  ) 
		ax12.set(xlabel='nominal - prediction'	, ylabel='ECM', title='Residues'	)

		if save: self.neural_regresor_model = model
		plt.show()


	def neural_regresor_predict(self, X:np.ndarray=None, Y:np.ndarray=None, Z:np.ndarray=None, N:np.ndarray=None, 
								epochs:int=1000,
								v:bool=True, save:bool=True):
		
		X 			= X 		if not X is None 		else self.X
		Y 			= Y 		if not Y is None 		else self.Y
		Z 			= Z 		if not Z is None 		else self.Z
		N 			= N 		if not N is None 		else self.N

		def coeff_determination(y_true, y_pred):
			SS_res = K.sum(K.square(y_true-y_pred))
			SS_tot = K.sum(K.square(y_true - K.mean(y_true)))

			return 1 - SS_res/SS_tot

		samples= X.shape[0]
		samples = Y.shape[0]

		from sklearn.neural_network import MLPRegressor
		from sklearn.model_selection import train_test_split

		import seaborn as sns
		sns.set_style('ticks')

		# === MODEL data === #
		print( '>> == Data shape ==' )
		print( f' (OK) X : {X.shape}' )
		print( f' (OK) Y : {Y.shape}' )
		print( f' (OK) N : {N.shape}' )

		# === load MODEL === #
		model = self.neural_regresor_model

		# === PLOT data === #
		# regresion ECM vs epoch #
		fig, ([ax11, ax12]) = plt.subplots(1, 2, figsize=(15, 5) )
		fig.tight_layout()

		# data vs PREDICT #
		y_regr = model.predict(X)
		ax11.plot(Y, y_regr, 'o', color=self.cnames['tomato'], label='validation data')
		ax11.set(xlabel='y data'	, ylabel='y prediction', title='Regresion Curve'	)
		ss_res = np.sum( (Y - model.predict(X) )**2  )	# Suma de los cuadrados de los residuos
		ss_tot = np.sum( (Y - np.mean(Y) )**2  ) 		# Suma total de cuadrados
		ax11.plot( [ 0.0, 5], [   0, 5.0] , ':' , alpha=0.5, c=(0,0,0), label=f'R2 = {1-(ss_res/ss_tot):.2}')
		ax11.plot( [ 0.0, 5], [ 0.5, 5.5] , '-' , alpha=0.5, c=(1,0,0), label='500meV error')
		ax11.plot( [ 0.0, 5], [-0.5, 4.5] , '-' , alpha=0.5, c=(1,0,0))
		ax11.legend(loc='best')

		#  RMSE hist  #
		ax12.hist( Y - model.predict(X)  ) 
		ax12.set(xlabel='nominal - prediction'	, ylabel='ECM', title='Residues'	)

		plt.show()

# ==== QM01 ==== # # ==== QM01 ==== # # ==== QM01 ==== ## ==== QM01 ==== # # ==== QM01 ==== # # ==== QM01 ==== #
# ==== QM01 ==== # # ==== QM01 ==== # # ==== QM01 ==== ## ==== QM01 ==== # # ==== QM01 ==== # # ==== QM01 ==== #
# ==== QM01 ==== # # ==== QM01 ==== # # ==== QM01 ==== ## ==== QM01 ==== # # ==== QM01 ==== # # ==== QM01 ==== #
qm1 = QMlearning()

# --> LOAD <-- # 
#qm1.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MTPyP_catalysis_part1')
#qm1.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MTPyP_catalysis_part2')
#qm1.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MPC_catalysis_part1')
#qm1.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MPC_catalysis_part2')
#qm1.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MTPyPwoAu_catalysis')
qm1.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MPCwoAu_catalysis')

# --> FILTER <-- # 
qm1.filter( filters=[{ 'type':'N', 'condition':'whitelist', 'list':['Sc','Ti','V','Cr','Mn',  'Fe','Co','Ni','Cu','Zn',] }] )
qm1.filter( filters=[{ 'type':'X', 'condition':'isNAN', }] )
qm1.filter( filters=[{ 'type':'Y', 'condition':'isNAN', }] )
qm1.filter( filters=[{ 'type':'N', 'condition':'whitelist', 'list':['D3', 'OPT'] }] )

# --> make METADA <-- # 
label = [ 	'overpotencial_ORR_4e', 'Gabs_OOH', 'Eabs_OOH', 'Gabs_OH',  'Eabs_OH', 'Gabs_O', 	'Eabs_O', 'G1_ORR', 	'G2_ORR', 'G3_ORR', 'G4_ORR',  'limiting_step_ORR']
color  = [ qm1.atomic_metadata[n[:2]]['color']  if n[:2] in qm1.atomic_metadata else qm1.atomic_metadata[n[:1]]['color']  if n[:1] in qm1.atomic_metadata else (1,0,0) for n in qm1.N ]
marker = [ qm1.functional_metadata[n]['marker'] if n in qm1.functional_metadata else '*' for n in [ n.split('_')[2] for n in qm1.N] ]

period_N, column_N, valence_N = [], [], []
for n in qm1.N:
	for key, metadata in qm1.atomic_metadata.items():
		if key in n[:2]: 
			period_N.append(metadata['period']), column_N.append(metadata['column']), valence_N.append(np.min(metadata['valence']))
			break;
period_N, column_N, valence_N = np.array(period_N), np.array(column_N), np.array(valence_N)
Ysplit = 500
def band_center(band, Emin=0, Emax=1 ):  return np.sum(band*np.linspace(Emin, Emax, num=band.shape[0]))/np.sum(band)
orbitals = [XYd, XYu, YZd, YZu, Z2d, Z2u, XZd, XZu, Y2d, Y2u] = [ qm1.Y[:, y*Ysplit+8:(y+1)*Ysplit+8] for y in range(10) ]
LCobitals = [	XYd+XYu,  YZd+YZu, 	 Z2d+Z2u, 	XZd+XZu, 	 Y2d+Y2u, 	 (XYd+XYu)+(Y2d+Y2u),   (YZd+YZu)+(XZd+XZu),   	 (YZd+YZu)+(XZd+XZu)+(Z2d+Z2u), 	XYd+XYu+YZd+YZu+Z2d+Z2u+XZd+XZu+Y2d+Y2u,]
LCnames =   [  'XYd+XYu','YZd+YZu', 'Z2d+Z2u', 'XZd+XZu', 	'Y2d+Y2u', 	'(XYd+XYu)+(Y2d+Y2u)', '(YZd+YZu)+(XZd+XZu)', 	'(YZd+YZu)+(XZd+XZu)+(Z2d+Z2u)', 	'XYd+XYu+YZd+YZu+Z2d+Z2u+XZd+XZu+Y2d+Y2u', 'period_N', 'column_N', 'valence_N'	]
BC = np.array([ [band_center(orbital[n, :], -5, 5)  for n in range(qm1.Y.shape[0])] for orbital in LCobitals]).T 
Y  = np.concatenate( (BC, np.array([period_N, column_N, valence_N]).T), axis=1)

# === Make DATAFRAME === # 
qm1.make_dataframe(qm1.X, LCnames, qm1.N)
qm1.dataframe_add(Y, label, qm1.N)

# --> Data analysis <-- # 
qm1.ORR_analysis(text=False, color=color, marker=marker, Y=Y, Ylabel=LCnames)

# --> NNmodel <-- # 
#qm1.neural_regresor_fit( X=Y, Y=qm1.X[:,0], epochs=1000 )
#qm1.deeplearnning_regresor_fit( X=qm1.Y, Y=qm1.X[:,1], epochs=200 )


# ==== QM s ==== # # ==== QM s ==== # # ==== QM s ==== # # ==== QM s ==== # # ==== QM s ==== # # ==== QM s ==== #
# ==== QM s ==== # # ==== QM s ==== # # ==== QM s ==== # # ==== QM s ==== # # ==== QM s ==== # # ==== QM s ==== #
# ==== QM s ==== # # ==== QM s ==== # # ==== QM s ==== # # ==== QM s ==== # # ==== QM s ==== # # ==== QM s ==== #
qm_s = QMlearning()

# --> LOAD <-- # 
qm_s.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/substitution/137cell/data')
occupancy = np.loadtxt('/home/akaris/Documents/code/VASP/v4.7/files/dataset/substitution/137cell/ocupation', delimiter=" ", dtype='str')

# --> FILTER <-- # 
qm_s.filter( filters=[{ 'type':'X', 'condition':'isNAN', }] )
qm_s.filter( filters=[{ 'type':'Y', 'condition':'isNAN', }] )
qm_s.filter( filters=[{ 'type':'X', 'condition':'minor', 'value':10}] )
#qm_s.filter( filters=[{ 'type':'N', 'condition':'whitelist', 'list':['OPTPBE'] }] )

# --> make METADA <-- # 
label = [ 	'overpotencial_ORR_4e', 'Gabs_OOH', 'Eabs_OOH', 'Gabs_OH',  'Eabs_OH', 'Gabs_O', 	'Eabs_O', 'G1_ORR', 	'G2_ORR', 'G3_ORR', 'G4_ORR',  'limiting_step_ORR']
color  = [ {'0':(1,0,0), '1':(0.8,0,0), '2':(0.6,0,0), '3':(0.4,0,0), '4':(0,1,0), '5':(0,0.8,0), '6':(0,0.6,0), '7':(0,0.4,0), '8':(0,0,1)}[n[7]] for n in qm_s.N ]
marker = [ qm_s.functional_metadata[n]['marker'] if n in qm_s.functional_metadata else '*' for n in [ n.split('_')[1] for n in qm_s.N] ]
color = [ qm_s.functional_metadata[n]['color'] if n in qm_s.functional_metadata else (0,0,0) for n in [ n.split('_')[1] for n in qm_s.N] ]
period_N, column_N, valence_N = np.array([ int(n[7]) for n in qm_s.N ]), np.array([ int(n[7]) for n in qm_s.N ]), np.array([ int(n[7]) for n in qm_s.N ])

Ysplit = 500
def band_center(band, Emin=0, Emax=1 ):  return np.sum(band*np.linspace(Emin, Emax, num=band.shape[0]))/np.sum(band)
orbitals = [XYd, XYu, YZd, YZu, Z2d, Z2u, XZd, XZu, Y2d, Y2u] = [ qm_s.Y[:, y*Ysplit+8:(y+1)*Ysplit+8] for y in range(10) ]
LCobitals = [	XYd+XYu,  YZd+YZu, 	 Z2d+Z2u, 	XZd+XZu, 	 Y2d+Y2u, 	 (XYd+XYu)+(Y2d+Y2u),   (YZd+YZu)+(XZd+XZu),   	 (YZd+YZu)+(XZd+XZu)+(Z2d+Z2u), 	XYd+XYu+YZd+YZu+Z2d+Z2u+XZd+XZu+Y2d+Y2u,]
LCnames =   [  'XYd+XYu','YZd+YZu', 'Z2d+Z2u', 'XZd+XZu', 	'Y2d+Y2u', 	'(XYd+XYu)+(Y2d+Y2u)', '(YZd+YZu)+(XZd+XZu', 	'(YZd+YZu)+(XZd+XZu)+(Z2d+Z2u)', 	'XYd+XYu+YZd+YZu+Z2d+Z2u+XZd+XZu+Y2d+Y2u', 'N1', 'N2', 'N3']
BC = np.array([ [band_center(orbital[n, :], -5, 5)  for n in range(qm_s.Y.shape[0])] for orbital in LCobitals]).T 
Y  = np.concatenate( (BC, np.array([period_N, column_N, valence_N]).T), axis=1)

filter_array = ~np.isnan(Y).any(axis=1) 
X = qm_s.X[filter_array,:]
Y = Y[filter_array,:]
N = qm_s.N[filter_array]
print( X.shape, Y.shape, N.shape )

# === Make DATAFRAME === # 
qm_s.make_dataframe(X, LCnames, N)
qm_s.dataframe_add(Y, label, N)

# --> Data analysis <-- # 
#qm_s.ORR_analysis(text=False, color=color, marker=marker, Y=Y, Ylabel=LCnames, occupancy=occupancy )

# --> NNmodel <-- # 
#qm_s.neural_regresor_model = qm1.neural_regresor_model
#qm_s.neural_regresor_predict( X=Y, Y=X[:,0] )
#qm_s.deeplearnning_predict( X=qm_s.Y, Y=qm_s.X[:,1], color=color, marker=marker)

# ==== QM02 ==== # # ==== QM02 ==== # # ==== QM02 ==== # # ==== QM02 ==== # # ==== QM02 ==== # # ==== QM02 ==== #
# ==== QM02 ==== # # ==== QM02 ==== # # ==== QM02 ==== # # ==== QM02 ==== # # ==== QM02 ==== # # ==== QM02 ==== #
# ==== QM02 ==== # # ==== QM02 ==== # # ==== QM02 ==== # # ==== QM02 ==== # # ==== QM02 ==== # # ==== QM02 ==== #
qm = QMlearning()

# --> LOAD <-- # 
qm.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/Bimetalic/M1TPyPM2FREE_catalysis_site2')
#qm.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/Bimetalic/M1TPyPM2Au_catalysis_site2')

qm.filter( filters=[{ 'type':'X', 'condition':'isNAN', }] )
qm.filter( filters=[{ 'type':'Y', 'condition':'isNAN', }] )

# --> make METADA <-- # 
color  = np.array([ qm.atomic_metadata[n[-2:]]['color']  if n[-2:] in qm.atomic_metadata else qm.atomic_metadata[n[-1:]]['color']  if n[-1:] in qm.atomic_metadata else (1,0,0) for n in qm.N ])
marker = np.array([ qm.atomic_metadata[n[:2]]['marker'] if n[:2] in qm.atomic_metadata else qm.atomic_metadata[n[:1]]['marker'] if n[:1] in qm.atomic_metadata else '*' for n in qm.N ])
#color  = np.array([ qm.atomic_metadata[n[:2]]['color']  if n[:2] in qm.atomic_metadata else qm.atomic_metadata[n[:1]]['color']  if n[:1] in qm.atomic_metadata else (1,0,0) for n in qm.N ])
#marker = np.array([ qm.atomic_metadata[n[-2:]]['marker'] if n[-2:] in qm.atomic_metadata else qm.atomic_metadata[n[-1:]]['marker'] if n[-1:] in qm.atomic_metadata else '*' for n in qm.N ])

# get M1
M1 = np.array([ n[:2]  if n[:2]  in qm.atomic_metadata else qm.atomic_metadata[n[:1]]['marker']  if n[:1]  in qm.atomic_metadata else '*' for n in qm.N ])
M2 = np.array([ n[-2:] if n[-2:] in qm.atomic_metadata else qm.atomic_metadata[n[-1:]]['marker'] if n[-1:] in qm.atomic_metadata else '*' for n in qm.N ])

period_N, column_N, valence_N = [], [], []
for n in qm.N:
	for key, metadata in qm.atomic_metadata.items():
		if key in n[:2] : 
			print(n, key)
			period_N.append(metadata['period']), column_N.append(metadata['column']), valence_N.append(np.min(metadata['valence']))
			break;
period_N, column_N, valence_N = np.array(period_N), np.array(column_N), np.array(valence_N)

Ysplit = 500
def band_center(band, Emin=0, Emax=1 ):  return np.sum(band*np.linspace(Emin, Emax, num=band.shape[0]))/np.sum(band)
orbitals = [XYd, XYu, YZd, YZu, Z2d, Z2u, XZd, XZu, Y2d, Y2u] = [ qm.Y[:, y*Ysplit+8:(y+1)*Ysplit+8] for y in range(10) ]
LCobitals = [	XYd+XYu,  YZd+YZu, 	 Z2d+Z2u, 	XZd+XZu, 	 Y2d+Y2u, 	 (XYd+XYu)+(Y2d+Y2u),   (YZd+YZu)+(XZd+XZu),   	 (YZd+YZu)+(XZd+XZu)+(Z2d+Z2u), 	XYd+XYu+YZd+YZu+Z2d+Z2u+XZd+XZu+Y2d+Y2u,]
LCnames =   [  'XYd+XYu','YZd+YZu', 'Z2d+Z2u', 'XZd+XZu', 	'Y2d+Y2u', 	'(XYd+XYu)+(Y2d+Y2u)', '(YZd+YZu)+(XZd+XZu)', 	'(YZd+YZu)+(XZd+XZu)+(Z2d+Z2u)', 	'XYd+XYu+YZd+YZu+Z2d+Z2u+XZd+XZu+Y2d+Y2u', 'period_N', 'column_N', 'valence_N'	]
BC = np.array([ [band_center(orbital[n, :], -5, 5)  for n in range(qm.Y.shape[0])] for orbital in LCobitals]).T 
Y  = np.concatenate( (BC, np.array([period_N, column_N, valence_N]).T), axis=1)
filter_array = ~np.isnan(Y).any(axis=1) 

# Filtrado de variables 2
# ==============================================================================
'''
Y=Y[filter_array,:]
N=qm.N[filter_array]
X=qm.X[filter_array, :] 
color = color[filter_array] 
marker = marker[filter_array] 
M1 = M1[filter_array] 
M2 = M2[filter_array] 
'''
X=qm.X
N=qm.N

# --> Data analysis <-- # 
#occupancy = np.loadtxt('/home/akaris/Documents/code/VASP/v4.7/files/dataset/Bimetalic/M1TPyPM2Au_catalysis_site1/ocupation_data', delimiter=" ", dtype='str')
#qm.ORR_analysis(text=False, color=color, marker=marker, X=X, Y=Y, N=N, Ylabel=LCnames) #, occupancy=occupancy )	#'names': ('system', 'occupancy', 'unoccupancy'),

# ---> Make DATAFRAME <--- # 
qm.make_dataframe(X, label, N)
#qm.dataframe_add(Y, LCnames, N)
print(qm.df)
qm.dataframe_add(M1, ['M1'], N)
qm.dataframe_add(M2, ['M2'], N)
qm.dataframe_add([qm.atomic_metadata_list[m][1] for m in M1], ['M1n'], N)
qm.dataframe_add([qm.atomic_metadata_list[m][1] for m in M2], ['M2n'], N)

import seaborn as sns
sns.scatterplot(data=qm.df, x="M2n", y="Gabs_OOH", hue="M1")
plt.show()

# make boxplot with Seaborn
bplot=sns.boxplot(y='Gabs_O', x='M1n', 
                 data=qm.df, 
                 width=0.5,
                 palette="colorblind")

# add stripplot to boxplot with Seaborn
bplot=sns.stripplot(y='overpotencial_ORR_4e', x='M1n', 
                   data=qm.df, 
                   jitter=True, 
                   marker='o', 
                   alpha=0.5,
                   color='black')

plt.show()

from scipy import stats
from scipy.stats import pearsonr

# https://www.cienciadedatos.net/documentos/pystats05-correlacion-lineal-python.html

#El diagrama de dispersión parece indicar una relación lineal positiva entre ambas variables.
# Para poder elegir el coeficiente de correlación adecuado, se tiene que analizar el tipo de variables y la distribución 
# que presentan. En este caso, ambas variables son cuantitativas continuas y pueden ordenarse para convertirlas en un ranking, 
# por lo que, a priori, los tres coeficientes podrían aplicarse. La elección se hará en función de la distribución que presenten 
# las observaciones: normalidad, homocedasticidad y presencia de outliers.

# Normalidad de los residuos Shapiro-Wilk test
# ==============================================================================
shapiro_test = stats.shapiro(qm.df['overpotencial_ORR_4e'])
print(f"Variable height: {shapiro_test}")
# El análisis gráfico y los test estadísticos muestran evidencias de que no se puede asumir normalidad en ninguna de las dos variables. 
# Siendo estrictos, este hecho excluye la posibilidad de utilizar el coeficiente de Pearson, dejando como alternativas el de Spearman o Kendall. 
# Sin embargo, dado que la distribución no se aleja mucho de la normalidad y de que el coeficiente de Pearson tiene cierta robustez, a fines 
# prácticos sí que se podría utilizar siempre y cuando se tenga en cuenta este hecho y se comunique en los resultados. Otra posibilidad es tratar 
# de transformar las variables para mejorar su distribución, por ejemplo, aplicando el logaritmo.

# Cálculo de correlación con Pandas
# ==============================================================================
print('Correlación Pearson: ', qm.df['overpotencial_ORR_4e'].corr(qm.df['XYd+XYu+YZd+YZu+Z2d+Z2u+XZd+XZu+Y2d+Y2u'], method='pearson'))
print('Correlación spearman: ', qm.df['overpotencial_ORR_4e'].corr(qm.df['XYd+XYu+YZd+YZu+Z2d+Z2u+XZd+XZu+Y2d+Y2u'], method='spearman'))
print('Correlación kendall: ', qm.df['overpotencial_ORR_4e'].corr(qm.df['XYd+XYu+YZd+YZu+Z2d+Z2u+XZd+XZu+Y2d+Y2u'], method='kendall'))

# Cálculo de correlación y significancia con Scipy
# ==============================================================================
r, p = stats.pearsonr(qm.df['overpotencial_ORR_4e'], qm.df['XYd+XYu+YZd+YZu+Z2d+Z2u+XZd+XZu+Y2d+Y2u'])
print(f"Correlación Pearson: r={r}, p-value={p}")

r, p = stats.spearmanr(qm.df['overpotencial_ORR_4e'], qm.df['XYd+XYu+YZd+YZu+Z2d+Z2u+XZd+XZu+Y2d+Y2u'])
print(f"Correlación Spearman: r={r}, p-value={p}")

r, p = stats.kendalltau(qm.df['overpotencial_ORR_4e'], qm.df['XYd+XYu+YZd+YZu+Z2d+Z2u+XZd+XZu+Y2d+Y2u'])
print(f"Correlación Pearson: r={r}, p-value={p}")
# Los test estadísticos muestran una correlación lineal entre moderada y alta, con claras evidencias estadísticas de que la relación 
# observada no se debe al azar (pvalue≈0). 

# Matriz de correlación	
# ==============================================================================
corr_matrix = qm.df.corr(method='spearman')

def tidy_corr_matrix(corr_mat):
    '''
    Función para convertir una matriz de correlación de pandas en formato tidy.
    '''
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1','variable_2','r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)
    
    return(corr_mat)

print(tidy_corr_matrix(corr_matrix).head(10))
# Heatmap matriz de correlaciones
# ==============================================================================
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
import seaborn as sns
sns.heatmap(
    corr_matrix,
    annot     = True,
    cbar      = False,
    annot_kws = {"size": 8},
    vmin      = -1,
    vmax      = 1,
    center    = 0,
    cmap      = sns.diverging_palette(20, 220, n=200),
    square    = True,
    ax        = ax
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation = 45,
    horizontalalignment = 'right',
)

ax.tick_params(labelsize = 10)
plt.show()


# Cálculo de correlación lineal
# ==============================================================================
import pingouin as pg
cc = pg.corr(x=qm.df['overpotencial_ORR_4e'], y=qm.df['XYd+XYu+YZd+YZu+Z2d+Z2u+XZd+XZu+Y2d+Y2u'], method='pearson')


# Cálculo de correlación lineal parcial
# ==============================================================================
pcc = pg.partial_corr(data=qm.df, x='overpotencial_ORR_4e', y='XYd+XYu+YZd+YZu+Z2d+Z2u+XZd+XZu+Y2d+Y2u', covar='(YZd+YZu)+(XZd+XZu)', method='pearson')
print(cc, pcc)



# BIBLIOGRAFIA
# Linear Models with R by Julian J.Faraway libro
#OpenIntro Statistics: Fourth Edition by David Diez, Mine Çetinkaya-Rundel, Christopher Barr libro
#Introduction to Machine Learning with Python: A Guide for Data Scientists libro
#Points of Significance: Association, correlation and causation. Naomi Altman & Martin Krzywinski Nature Methods

# Cálculo ANOVA
# ==============================================================================
import scipy.stats as stats

ftest = stats.f_oneway(qm.df['overpotencial_ORR_4e'][qm.df['M2'] == 'Fe'],
               qm.df['overpotencial_ORR_4e'][qm.df['M2'] == 'Co'],
               qm.df['overpotencial_ORR_4e'][qm.df['M2'] == 'Sc'],
               qm.df['overpotencial_ORR_4e'][qm.df['M2'] == 'Cr'],
               qm.df['overpotencial_ORR_4e'][qm.df['M2'] == 'Ti'],
               qm.df['overpotencial_ORR_4e'][qm.df['M2'] == 'Ni'],
               )

print(ftest)

# Cálculo MANOVA
# ==============================================================================
from statsmodels.multivariate.manova import MANOVA
fit = MANOVA.from_formula('overpotencial_ORR_4e  + M2 ~ M2 + M1', data=qm.df)
print(fit.mv_test())


import seaborn as sns
sns.set_theme(style="whitegrid")
sns.barplot(data=qm.df, x="M2", y='XYd+XYu+YZd+YZu+Z2d+Z2u+XZd+XZu+Y2d+Y2u', hue="M1" )
plt.show()

# --> NNmodel <-- # 
qm.neural_regresor_model = qm1.neural_regresor_model
#qm.neural_regresor_predict( X=Y, Y=X )
qm.deeplearnning_predict( X=Y, Y=X, color=color, marker=marker)


asddassad








# ==== QM00 ==== # # ==== QM00 ==== # # ==== QM00 ==== #
qm = QMlearning()

# --> LOAD <-- # 
qm.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MPC_catalysis_part1')
qm.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MPC_catalysis_part2')
qm.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MTPyPwoAu_catalysis')

#qm.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MTPyP_catalysis_part1')
#qm.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MTPyP_catalysis_part2')
#qm.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MPCwoAu_catalysis')

# --> FILTER <-- # 
qm.filter( filters=[{ 'type':'N', 'condition':'whitelist', 'list':['Sc','Ti','V','Cr','Mn',  'Fe','Co','Ni','Cu','Zn',] }] )
qm.filter( filters=[{ 'type':'X', 'condition':'isNAN', }] )
qm.filter( filters=[{ 'type':'Y', 'condition':'isNAN', }] )
qm.filter( filters=[{ 'type':'N', 'condition':'whitelist', 'list':['D3', 'OPT'] }] )

# --> NNmodel <-- # 
qm.deeplearnning_regresor_fit( X=qm.Y, Y=qm.X, epochs=200 )

# --> Data analysis <-- # 
#qm.ORR_analysis(text=False)

# ==== QM01 ==== # # ==== QM01 ==== # # ==== QM01 ==== #
qm1 = QMlearning()

# --> LOAD <-- # 
#qm1.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MPC_catalysis_part1')
#qm1.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MPC_catalysis_part2')
#qm1.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MTPyPwoAu_catalysis')

qm1.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MTPyP_catalysis_part1')
qm1.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MTPyP_catalysis_part2')
#qm1.load_folder('/home/akaris/Documents/code/VASP/v4.7/files/dataset/metales/MPCwoAu_catalysis')

# --> FILTER <-- # 
qm1.filter( filters=[{ 'type':'N', 'condition':'whitelist', 'list':['Sc','Ti','V','Cr','Mn',  'Fe','Co','Ni','Cu','Zn',] }] )
qm1.filter( filters=[{ 'type':'X', 'condition':'isNAN', }] )
qm1.filter( filters=[{ 'type':'Y', 'condition':'isNAN', }] )
qm1.filter( filters=[{ 'type':'N', 'condition':'whitelist', 'list':['D3', 'OPT'] }] )

# --> NNmodel <-- # 
qm1.neural_regresor_model = qm.neural_regresor_model
qm1.deeplearnning_predict( X=qm1.Y, Y=qm1.X )

plt.show()




