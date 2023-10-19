# *** warning supresion
import warnings, os, argparse, datetime
warnings.filterwarnings("ignore")
import warnings; warnings.filterwarnings("ignore")

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

class KPOINTS(object):
	def __init__(self, file_name:str=None, path:str=None, name:str=None,
						comment:str=None, kpoints_number:int=None, mesh_center:str=None, subdivisions:list=None, shift:list=None ):
		self.name 			= name
		self.path 			= path 		
		self.file_name 		= file_name

		'''
		Point     Cartesian coordinates     Fractional coordinates
		            (units of 2pi/a)         (units of b1,b2,b3)
		----------------------------------------------------------
		  Γ         (  0    0    0  )         (  0    0    0  )
		  X         (  0    0    1  )         ( 1/2  1/2   0  )
		  W         ( 1/2   0    1  )         ( 1/2  3/4  1/4 )
		  K         ( 3/4  3/4   0  )         ( 3/8  3/8  3/4 )
		  L         ( 1/2  1/2  1/2 )         ( 1/2  1/2  1/2 )
		'''
		self.G = [0.0, 0.0, 0.0]
		self.X = [0.0, 0.0, 1.0]
		self.W = [1/2, 0.0, 1.0]
		self.K = [3/4, 3/4, 0.0]
		self.L = [1/2, 1/2, 1/2]


		self.comment 		= comment			
		# (LINE 1) The first line is a comment line
		
		self.kpoints_number = kpoints_number 	
		# (LINE 2) In the second line, set the number of k points to 0 to indicate an automatic mesh generation.
		# kpoints_number :: [int] :: 0 -> automesh

		self.mesh_center	= mesh_center	
		# (LINE 3) The first nonblank character of the third line determines the center of the mesh. The possible choice are Γ-centered (G, g) or the Monkhorst-Pack
		# Gamma-centered 		<<<< (G, g)
		# Monkhorst-Pack scheme <<<< (M, m)

		self.subdivisions 	= subdivisions
		# (LINE 4) Specify the desired number of subdivisions N1, N2, and N3 in the fourth line
		# subdivisions ::  >>>> 0 0 0 (gamma-only)

		self.shift 			= shift
		# (LINE 5) Optionally add a fifth line to shift the mesh by ( S1, S2, and S3) with respect to the default.
		# shift :: [int, int, int] >>>> 0 0 0 no shift

		self.text 			= None
		# store the full-plain text

		self.path = None

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

	def isnum(self, n:int):
		'''
		 ------------------ Define if n is or not a number ------------------ # 
		 n     :   VAR     :   VAR to check if it is a numerical VAR
		 return :  BOOL    : True/False
		1.      The float() function takes a single parameter, n, which is the number to be converted.
		2.      If n is not a number, the function returns False.
		3.      If n is a number, the function returns the float() converted value.
		'''
		try: float(n); return True
		except: return False
	
	@Logs.LogDecorator()
	def load(self, 	file:str=None, meshes_type:str='regular',
					save:bool=True, verbosity:bool=True):
		if file is None:  file = self.file_name 

		if 		meshes_type == 'regular':
			self.load_regular(file=file, save=save, verbosity=verbosity)
		elif 	meshes_type == 'generalized':
			self.load_generalized(file=file, save=save, verbosity=verbosity)
		
		if save: self.file_name = file

		return True

	@Logs.LogDecorator()
	def load_regular(self, 	file:str=None, meshes_type:str='regular',
							save:bool=True, verbosity:bool=True):
		file = open(f'{file}')
		lines = [None for n in range(5) ] 

		for i, line in enumerate(file):
			vec = [float(m) if self.isnum(m) else m for m in line.split(' ') if m != '']
			if i == 0: 		lines[i] = line.replace("\n", "") 
			if i == 1: 		lines[i] = vec[0] 
			if i == 2: 		lines[i] = line.replace("\n", "") 

			if i > 2 and (lines[2].strip().upper()[0] == 'M' or lines[2].strip().upper()[0] == 'G'):
				lines[i] = vec 
			elif i > 2:
				lines[i] = line.replace("\n", "") 

		if save:
			try:
				self.comment, self.kpoints_number, self.mesh_center, self.subdivisions, self.shift = lines
				self.text = ''.join(lines)
			except: print(f'ERROR :: KPOINTS.load_regular() :: can NOT load {file}')

	@Logs.LogDecorator()
	def print(self, ):
		print(f'>> comment        :: {self.comment}')
		print(f'>> kpoints_number :: {self.kpoints_number}')
		print(f'>> mesh_center    :: {self.mesh_center}')
		print(f'>> subdivisions   :: {self.subdivisions}')
		print(f'>> shift          :: {self.shift}')
		print('-'*40)
		print(self.text)

		return True

	def summary(self, ):		self.print()
	def view(self, ):			self.print()

	def save(self, name:str=None):
		with open(str(name), 'w') as kpoint_file:
			kpoint_file.write(f'{datetime.datetime.now()} :: {self.comment}\n')
			kpoint_file.write(f'{self.kpoints_number}\n')
			kpoint_file.write(f'{self.mesh_center}\n')
			if type(self.subdivisions) == str:	kpoint_file.write(f'{self.subdivisions}\n')
			else:								kpoint_file.write(f'{self.subdivisions[0]}  {self.subdivisions[1]}  {self.subdivisions[2]}\n')

			if type(self.shift) == str:	kpoint_file.write(f'{self.shift}\n')
			else:						kpoint_file.write(f'{self.shift[0]}  {self.shift[1]}  {self.shift[2]}\n')

	def load_default(self, content='MP'):
		# MP >> Monkhorst-Pack mesh  
		if content == 'MP':
			self.comment    	= ''
			self.kpoints_number = '0'
			self.mesh_center 	= 'Monkhorst-Pack mesh'
			self.subdivisions 	= np.array([1,1,1])
			self.shift 			= np.array([0,0,0])

	def kpoints_optimization(self, kpoints:list=None, path:str=None):
		self.load_default(content='MP')
		os.makedirs(f'{path}/kpoints_optimization', exist_ok=True)
		for i, k in enumerate(kpoints):
			self.subdivisions = k 
			os.makedirs(f'{path}/kpoints_optimization/{i}-{k[0]}_{k[1]}_{k[2]}', exist_ok=True)
			self.save(f'{path}/kpoints_optimization/{i}-{k[0]}_{k[1]}_{k[2]}/KPOINTS')

if __name__ == "__main__":
	kpoints = KPOINTS()
	kpoints.load('/home/akaris/Documents/code/VASP/v4.7/files/KPOINTS/KPOINTS')
	kpoints.save('/home/akaris/Documents/code/VASP/v4.7/files/KPOINTS/KPOINTS2')
	kpoints.kpoints_optimization(   kpoints = [ [n,n,n] for n in range(30) ], 
									path='/home/akaris/Documents/code/VASP/v4.7/files/KPOINTS' )
