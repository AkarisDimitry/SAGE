# *** warning supresion
import warnings, os
from numba import njit, jit
warnings.filterwarnings("ignore")
from math import cos, sin
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

from auxiliary import *

class LCAO(object):
	def __init__(self, file:str=None, N:int=None, L:int=None):
		self.L  = L
		self.POSCAR = None
		self.V = 5 # neighboring cell to consider

		self.hopping = None
		self.parameters = [[[ [1,1,1,5],[1,1,1,5],[1,1,1,5],[1,1,1,5],[1,1,1,5],[1,1,1,5],[1,1,1,5],[1,1,1,5],[1,1,1,5],[1,1,1,5], ]]]
		self.Phi = [sss, sps, pps, ppp, sds, pds, pdp, dds, ddp, ddd]

		# The Slater-Koster hopping integrals ~in eV! at the
		# bulk fcc equilibrium interatomic spacing R0 and the corresponding
		# parameter qb governing their variation with distance @Eq.3 
		# DIM :: SPD X SPD X SPD :: 3x3x3
		#				Xss		Xsp		Xsd			Xps		Xpp		Xpd			Xds		Xdp		Xdd
		self.B0 = [ [ [-0.9755, 0.0000, 0.0000], [ 1.9945, 0.0000, 0.0000], [-0.9488, 0.0000, 0.0000] ], # Sxx
					[ [ 1.9945, 0.0000, 0.0000], [ 3.3313,-0.1218, 0.0000], [-1.3096, 0.1561, 0.0000] ], # Pxx
					[ [-0.9488, 0.0000, 0.0000], [-1.3096, 0.1561, 0.0000], [-0.9132, 0.5176,-0.0806] ] ]# Dxx
		#				Xss		Xsp		Xsd			Xps		Xpp		Xpd			Xds		Xdp		Xdd
		self.qb = [ [ [ 2.2556, 0.0000, 0.0000], [ 2.9709, 0.0000, 0.0000], [ 2.6455, 0.0000, 0.0000] ], # Sxx
					[ [ 2.9709, 0.0000, 0.0000], [ 3.1266, 3.1266, 0.0000], [ 3.7810, 3.7810, 0.0000] ], # Pxx
					[ [ 2.6455, 0.0000, 0.0000], [ 3.7810, 3.7810, 0.0000], [ 4.8526, 4.8526, 4.8526] ] ]# Dxx

		# (sss,sps,sds,pps,ppp,pds,pdp,dds,ddp,ddd)

		# TABLE I. The parameters ~in eV! determining the on-site s,p,d
		# matrix elements of the tight-binding Hamiltonian for rhodium. The
		# value of p in Eq. ~7! is p59.5270. The reference energy of these
		# matrix elements has been chosen such that the total energy per bulk
		# atom at equilibrium in the fcc structure is the opposite of the ex-
		# perimental cohesive energy ~5.78 eV!.
		self.a = [ 1.3329, 6.1674, 6.1674, 6.1674, 0.1259, 0.1259, 0.1259, 0.1259, 0.1259 ]
		self.b = [ 3.0637, 2.9454, 2.9454, 2.9454,-0.0370,-0.0370,-0.0370,-0.0370,-0.0370 ]
		self.c = [-0.1788,-0.2247,-0.2247,-0.2247, 0.0233, 0.0233, 0.0233, 0.0233, 0.0233 ]
		self.d = [ 0.0069, 0.0094, 0.0094, 0.0094, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009 ]

		# Two atomic structures ~fcc and bcc! have been considered, and the 
		# interatomic spacing has been varied between 0.9R0 and 1.1R0 ,
		# where R0 is the equilibrium fcc bulk interatomic distance. A
		# small number of large interatomic spacings up to 1.5R0 has
		# also been included in the fit in order to avoid an unphysical
		# behavior of the on-site parameters at long range. The param-
		# eters of the cutoff function have been fixed to D50.15 Å
		# and Rc55.5 Å, i.e., the hopping integrals become very small
		# beyond the fourth neighbors.
		self.delta = 0.15 # angstrom
		self.R0 = 1.0 # angstrom
		self.p  = 1.0 # angstrom
		self.Rc = 5.5 # angstrom

	def load(self, file:str=None):
		self.POSCAR = POSCAR.POSCAR()
		self.POSCAR.load(file)

		self.hopping = np.zeros((self.POSCAR.N*9, self.POSCAR.N*9))

		return True

	def get_v(self, t, p):
		r3 = 3**0.5		
		ct, st, cp, sp = cos(t), sin(t), cos(p), sin(p)

		return [[                     1,                 0,              0,              0,                         0],

				[                 st*cp,             ct*cp,             sp,              0,                         0],
				[                 st*sp,             ct*sp,            -cp,              0,                         0],
				[                    ct,               -st,              0,              0,                         0],

				[	     r3*st*st*sp*cp,     2*st*ct*sp*cp, st*(1-2*cp*cp), ct*(1-2*cp*cp),	          sp*cp*(st*st-2)],
				[	        r3*st*ct*sp,  2*st*(2*ct*ct-1),         -ct*cp,          st*cp,                  st*ct*sp],
				[		    r3*st*ct*cp,  2*cp*(2*ct*ct-1),          ct*st,         -st*sp,                  st*ct*cp],
				[r3/2*st*st*(2*cp*cp-1), st*ct*(2*cp*cp-1),     2*st*sp*cp,     2*ct*sp*cp, (st*st*0.5-1)*(2*cp*cp-1)],
				[       ct*ct-0.5*st*st,         -r3*st*ct,              0,              0,             -r3*0.5*st*st],
				]

	def get_rho(self, p:float=None, R0:float=None, V:int=None):
		p     = p   if not p  is None else self.p
		R0    = R0  if not R0 is None else self.R0
		V     = V   if not V  is None else self.V

		N = self.POSCAR.N
		cell = self.POSCAR.cell
		rho = np.zeros(N) # base

		R = [ [int(n1),int(n2),int(n3)] for n1 in range(-V,V+1) for n2 in range(-V,V+1) for n3 in range(-V,V+1)]

		for i in range(N):
			Ri = self.POSCAR.atoms[i,:]
			for Rn in R:
				for j in range(N):
					if not (Rn == [0,0,0] and i != j):
						Rj = self.POSCAR.atoms[j,:] + (cell[0,:]*Rn[0] + cell[1,:]*Rn[1] + cell[2,:]*Rn[2])
						Rij = Ri - Rj
						Rija = np.sqrt(sum([v**2 for v in Rij]))
						rho[i] += np.e**(-p*(Rija/R0 - 1)) * self.get_fc(Rija)

		return rho

	def get_fc(self, R, Rc:float=None, delta:float=None):
		'''
		Finally f c(R) is a cutoff function
		Rc and D determine, respectively, the cutoff distance and the
		steepness of the cutoff.
		# === PLOT === #
		R = np.arange(0,10,0.1)
		plt.plot(R, (1 + np.e**((R-Rc)/delta) )**-1)
		plt.show()
		'''
		delta = delta if not delta is None else self.delta
		Rc    = Rc    if not Rc    is None else self.Rc

		return (1 + np.e**((R-Rc)/delta) )**-1

	def get_epsilon(self, a:list=None, b:list=None, c:list=None, d:list=None):
		a     = a   if not a  is None else self.a
		b     = b   if not b  is None else self.b
		c     = c   if not c  is None else self.c
		d     = d   if not d  is None else self.d

		rho = self.get_rho()
		N = self.POSCAR.N
		epsilon = np.zeros((N, 9)) # base X orbitales 
		for i in range(N):
			for orbital in range(9):
				epsilon[i, orbital] = a[orbital] + b[orbital]*rho[i]**(2/3) + c[orbital]*rho[i]**(4/3) + d[orbital]*rho[i]**(2)

		return epsilon

	def get_direction_cosines(self, vector):
		return [v / np.sqrt(sum([v**2 for v in vector])) for v in vector]

	def get_spherical_coordinates(self, v):
		vxy = v[0]**2 + v[1]**2
		r = np.sqrt(vxy + v[2]**2)
		t = np.arctan2(np.sqrt(vxy), v[2]) 
		p = np.arctan2(v[1], v[0])
		return r, t, p

	def get_T(self, a:int, b:int, M:int, v:list=None):
		Mp = [0,1,3][M]
		return v[a][Mp]*v[b][Mp] + (1-int(M==0))*v[a][Mp+1]*v[b][Mp+1]

	def get_B(self, a:int, b:int, M:int, v:list=None, B0:float=None, qb:float=None, R0:float=None, R:float=None):
		B0     = B0  if not B0  is None else self.B0
		qb     = qb  if not qb  is None else self.qb
		R0     = R0  if not R0  is None else self.R0
		bp = [0,1,1,1,2,2,2,2,2][b]
		ap = [0,1,1,1,2,2,2,2,2][a]

		return B0[ap][bp][M] * np.e**(-qb[ap][bp][M]*((R/R0)-1) ) * self.get_fc(R)

	@jit( nopython=True) # , nopython=True, 
	def get_hopping(self, a:int=None, b:int=None, R:float=None, v:list=None):
		I = 0
		bp = [0,1,1,1,2,2,2,2,2][b]
		ap = [0,1,1,1,2,2,2,2,2][a]

		if ap<bp:
			for M in range(ap+1):	
				I += self.get_T(a=a, b=b,M=M, v=v) * self.get_B(a=a, b=b,M=M,v=v, R=R)
		else:
			for M in range(bp+1):	
				I += self.get_T(a=a, b=b,M=M, v=v) * self.get_B(a=a, b=b,M=M,v=v, R=R)

		return I

	def get_matrix_element(self, k, i, a, V:int=None):
		V     = V   if not V  is None else self.V
		Ri = self.POSCAR.atoms[i,:]
		N = self.POSCAR.N
		cell = self.POSCAR.cell

		R = [ [int(n1),int(n2),int(n3)] for n1 in range(-V,V+1) for n2 in range(-V,V+1) for n3 in range(-V,V+1)]
		R = [ [int(n1),int(n2),int(n3)] for n1 in range(-V,V+1) for n2 in range(-V,V+1) for n3 in range(-V,V+1)]
		for Rn in R:
			for b in range(9):
				for j in range(N):
					Rj = self.POSCAR.atoms[j,:] + (cell[0,:]*Rn[0] + cell[1,:]*Rn[1] + cell[2,:]*Rn[2])
					Rij = Ri - Rj
					Rija = np.sqrt(sum([v**2 for v in Rij]))
					r, t, p = self.get_spherical_coordinates( Rij )
					v = self.get_v(t, p) # orbitals, 5

					e1  = np.e**(np.dot(k,Rn)*1j)
					I = self.get_hopping(a=a, b=b, v=v, R=Rija)
					O = self.epsilon[i,a]*int(Rija==0) + (1-int(Rija==0))*I
					D = 1 + int(i==j) * int(Rija==0) * (int(a==b)-1)

					self.H[i*9+a][j*9+b] +=  e1*O*D
		return  

	@Logs.LogDecorator()
	def get_eigenvalues(self, k):
		
		self.epsilon = self.get_epsilon()
		N = self.POSCAR.N
		self.H = np.zeros((N*9, N*9))
		for i in range(N):
			for a in range(9):
				
				self.get_matrix_element(k, i, a)

		diag = np.linalg.eig( self.H )
		print( diag[0] )
		return self.H

	def two_center_integrals(M:np.ndarray=None):
		pass

path = '/home/akaris/Documents/code/VASP/v4.7/files/bulk_optimization/Pt/initial_files/POSCAR_FCC'
Rh = LCAO()
Rh.load(path)
Rh.get_eigenvalues(k=[0,0,0])




