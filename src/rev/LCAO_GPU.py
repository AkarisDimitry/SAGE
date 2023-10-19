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

@jit('complex128[:](complex128[:,:])', nopython=True) 
def diagonal_GPU(H):
	diag = np.linalg.eig( H )[0]
	return diag

@jit('float64[:,:](float64, float64)', nopython=True) 
def get_v(t, p):
	'''
	r3 = 3**0.5	
	matrix =  np.array([[ 1.2, 0, 0, 0, 0],[ 1, 0, 0, 0, 0]])
	return r3
	'''
	r3 = 3**0.5		
	ct, st, cp, sp = cos(t), sin(t), cos(p), sin(p)
	ct = 0 if ct < 10**-10 else ct
	st = 0 if st < 10**-10 else st
	cp = 0 if cp < 10**-10 else cp
	sp = 0 if sp < 10**-10 else sp

	M = np.zeros((9,5), dtype=np.float64)
	M[0,:] = [                     1,                 0,              0,              0,                         0]

	M[1,:] = [                 st*cp,             ct*cp,             sp,              0,                         0]
	M[2,:] = [                 st*sp,             ct*sp,            -cp,              0,                         0]
	M[3,:] = [                    ct,               -st,              0,              0,                         0]

	M[4,:] = [	     r3*st*st*sp*cp,     2*st*ct*sp*cp, st*(1-2*cp*cp), ct*(1-2*cp*cp),	          sp*cp*(st*st-2)]
	M[5,:] = [	        r3*st*ct*sp,  2*st*(2*ct*ct-1),         -ct*cp,          st*cp,                  st*ct*sp]
	M[6,:] = [		    r3*st*ct*cp,  2*cp*(2*ct*ct-1),          ct*st,         -st*sp,                  st*ct*cp]
	M[7,:] = [r3/2*st*st*(2*cp*cp-1), st*ct*(2*cp*cp-1),     2*st*sp*cp,     2*ct*sp*cp, (st*st*0.5-1)*(2*cp*cp-1)]
	M[8,:] = [       ct*ct-0.5*st*st,         -r3*st*ct,              0,              0,             -r3*0.5*st*st]

	return M

@jit('float64[:](float64[:])', nopython=True) 
def get_spherical_coordinates( v):
	vxy = v[0]**2 + v[1]**2
	M = np.zeros(3, dtype=np.float64)
	# === RHO === #   r = np.sqrt(x**2 + y**2 + z**2)
	M[0] = np.sqrt(vxy + v[2]**2)
	# === THETA === #   theta = math.acos(z / r)
	M[1] = np.arctan2(np.sqrt(vxy), v[2])  
	# === PHI === #    phi = math.atan2(y, x)
	M[2] = np.arctan2(v[1], v[0]) # phi

	return M

@jit('float64(int64, int64, float64, float64[:,:], float64, float64, float64, float64[:,:,:], float64[:,:,:])', nopython=True) 
def get_hopping(a, b, R, v, Rc, R0, delta, B0, qb):
	I = 0.0
	bp = 0 if b == 0 else 1 if b < 4 else 2  #[0,1,1,1,2,2,2,2,2][b]
	ap = 0 if a == 0 else 1 if a < 4 else 2  #[0,1,1,1,2,2,2,2,2][a]

	fc = (1 + np.e**((R-Rc)/delta) )**-1
	#fc = 1 if R < R0*1.11 else 0 
	if ap<bp:
		for M in range(ap+1):	
			Mp = M if M < 2 else 3 # [0,1,3][M]
			T = v[a][Mp]*v[b][Mp] + (1-int(M==0))*v[a][Mp+1]*v[b][Mp+1]
			B = B0[ap][bp][M] * np.e**(-qb[ap][bp][M]*((R/R0)-1) ) * fc
			I += T * B
	else:
		for M in range(bp+1):	
			Mp = M if M < 2 else 3 # [0,1,3][M]
			T = v[a][Mp]*v[b][Mp] + (1-int(M==0))*v[a][Mp+1]*v[b][Mp+1]
			B = B0[ap][bp][M] * np.e**(-qb[ap][bp][M]*((R/R0)-1) ) * fc
			I += T * B

	return I


@jit('complex128[:,:](float64[:], float64[:,:], float64[:,:], float64[:,:], int64, int64, float64, float64,  float64, float64[:,:,:], float64[:,:,:])', nopython=True) 
def get_H( k, epsilon, atoms, cell, N, V, Rc, R0, delta, B0, qb):
		
		H = np.zeros((N*9, N*9), dtype=np.complex128)
		#R = [ [int(n1),int(n2),int(n3)] for n1 in range(-V,V+1) for n2 in range(-V,V+1) for n3 in range(-V,V+1)]
		R = [ cell[0,:]*int(n1)+cell[1,:]*int(n2)+cell[2,:]*int(n3) for n1 in range(-V,V+1) for n2 in range(-V,V+1) for n3 in range(-V,V+1)]
		for i in range(N):
			Ri = atoms[i,:] 
			for Rn in R:
				e1  = np.e**( (k[0]*Rn[0]+k[1]*Rn[1]+k[2]*Rn[2])*1j)
				for j in range(N):
					Rj = atoms[j,:] + Rn

					Rij = Rj - Ri
					Rija = np.sqrt(Rij[0]*Rij[0]+Rij[1]*Rij[1]+Rij[2]*Rij[2])
					r, t, p = get_spherical_coordinates( Rij )

					v = get_v( float(t), float(p) ) # orbitals, 5
					DRn = int(Rn[0]==0)*int(Rn[1]==0)*int(Rn[2]==0)

					for a in range(9):
						for b in range(9):
							I = get_hopping(a=int(a), b=int(b), R=float(Rija), v=v, Rc=Rc, R0=R0, delta=delta, B0=B0, qb=qb)

							O = epsilon[i,a]*int(Rija==0)*int(a==b) + (1-int(Rija==0))*I
							D = 1 + int(i==j) * DRn * (int(a==b)-1)

							H[i*9+a][j*9+b] +=  e1*O*D

		return H

class LCAO(object):
	def __init__(self, file:str=None, N:int=None, L:int=None):
		self.L  = L
		self.POSCAR = None
		self.V = 5 # neighboring cell to consider

		self.hopping = None

		# The Slater-Koster hopping integrals ~in eV! at the
		# bulk fcc equilibrium interatomic spacing R0 and the corresponding
		# parameter qb governing their variation with distance @Eq.3 
		# DIM :: SPD X SPD X SPD :: 3x3x3
		#				Xss		Xsp		Xsd			Xps		Xpp		Xpd			Xds		Xdp		Xdd
		self.B0 = np.array([ 
					[ [-0.9755, 0.0000, 0.0000], [ 1.9945, 0.0000, 0.0000], [-0.9488, 0.0000, 0.0000] ], # Sxx
					[ [ 1.9945, 0.0000, 0.0000], [ 3.3313,-0.1218, 0.0000], [-1.3096, 0.1561, 0.0000] ], # Pxx
					[ [-0.9488, 0.0000, 0.0000], [-1.3096, 0.1561, 0.0000], [-0.9132, 0.5176,-0.0806] ]  # Dxx
					], dtype=np.float64 )
		#				Xss		Xsp		Xsd			Xps		Xpp		Xpd			Xds		Xdp		Xdd
		self.qb = np.array([ 
					[ [ 2.2556, 0.0000, 0.0000], [ 2.9709, 0.0000, 0.0000], [ 2.6455, 0.0000, 0.0000] ], # Sxx
					[ [ 2.9709, 0.0000, 0.0000], [ 3.1266, 3.1266, 0.0000], [ 3.7810, 3.7810, 0.0000] ], # Pxx
					[ [ 2.6455, 0.0000, 0.0000], [ 3.7810, 3.7810, 0.0000], [ 4.8526, 4.8526, 4.8526] ]  # Dxx
					], dtype=np.float64 )

		# (sss,sps,sds,pps,ppp,pds,pdp,dds,ddp,ddd)

		# TABLE I. The parameters ~in eV! determining the on-site s,p,d
		# matrix elements of the tight-binding Hamiltonian for rhodium. The
		# value of p in Eq. ~7! is p59.5270. The reference energy of these
		# matrix elements has been chosen such that the total energy per bulk
		# atom at equilibrium in the fcc structure is the opposite of the ex-
		# perimental cohesive energy ~5.78 eV!.
		self.a = np.array([ 1.3329, 6.1674, 6.1674, 6.1674, 0.1259, 0.1259, 0.1259, 0.1259, 0.1259 ])
		self.b = np.array([ 3.0637, 2.9454, 2.9454, 2.9454,-0.0370,-0.0370,-0.0370,-0.0370,-0.0370 ])*0
		self.c = np.array([-0.1788,-0.2247,-0.2247,-0.2247, 0.0233, 0.0233, 0.0233, 0.0233, 0.0233 ])*0
		self.d = np.array([ 0.0069, 0.0094, 0.0094, 0.0094,-0.0009,-0.0009,-0.0009,-0.0009,-0.0009 ])*0
		# Two atomic structures ~fcc and bcc! have been considered, and the 
		# interatomic spacing has been varied between 0.9R0 and 1.1R0 ,
		# where R0 is the equilibrium fcc bulk interatomic distance. A
		# small number of large interatomic spacings up to 1.5R0 has
		# also been included in the fit in order to avoid an unphysical
		# behavior of the on-site parameters at long range. The param-
		# eters of the cutoff function have been fixed to D50.15 Å
		# and Rc55.5 Å, i.e., the hopping integrals become very small
		# beyond the fourth neighbors.
		self.delta = 0.15   # angstrom
		self.p     = 9.69   # angstrom
		self.R0    = 2.8284271247461903 # angstrom
		self.Rc    = 5.5    # angstrom

	def load(self, file:str=None):
		self.POSCAR = POSCAR.POSCAR()
		self.POSCAR.load(file)

		self.hopping = np.zeros((self.POSCAR.N*9, self.POSCAR.N*9))

		return True

	def get_rho(self, p:float=None, R0:float=None, V:int=None):
		p     = p   if not p  is None else self.p
		R0    = R0  if not R0 is None else self.R0
		V     = V   if not V  is None else self.V

		N = self.POSCAR.N
		cell = self.POSCAR.cell*self.POSCAR.scale
		atoms = self.POSCAR.atoms*self.POSCAR.scale
		rho = np.zeros(N) # base

		R = [ cell[0,:]*int(n1)+cell[1,:]*int(n2)+cell[2,:]*int(n3) for n1 in range(-V,V+1) for n2 in range(-V,V+1) for n3 in range(-V,V+1)]
		for i in range(N):
			Ri = atoms[i,:]
			for Rn in R:
				for j in range(N):
					if not (Rn[0] == 0 and Rn[1] == 0 and Rn[2] == 0 and i == j):
						Rj = atoms[j,:] + Rn
						Rij = Rj - Ri
						Rija = np.sqrt([Rij[0]*Rij[0]+Rij[1]*Rij[1]+Rij[2]*Rij[2]])
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
		#return 1 if R < self.R0*1.1 else 0

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


	#@Logs.LogDecorator()
	def get_eigenvalues(self, k):
		k = np.array(k, dtype=np.float64)
		epsilon = self.get_epsilon()
		N = self.POSCAR.N
		atoms = self.POSCAR.atoms*self.POSCAR.scale
		cell = self.POSCAR.cell*self.POSCAR.scale

		V = self.V
		Rc, R0, delta, B0, qb = self.Rc, self.R0, self.delta, self.B0,self.qb
		H = get_H(k=k, epsilon=epsilon, atoms=atoms, cell=cell, N=N, V=V, Rc=Rc, R0=R0, delta=delta, B0=B0, qb=qb)

		eigenvalues = diagonal_GPU(H)

		return eigenvalues

	@Logs.LogDecorator()
	def BANDS(self, ):
		data = []
		N = 50

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_aspect('equal')
		ax.set_xlabel('KPOINT')
		ax.set_xlabel('E(eV)')

		ax.set_xticks([0,N])
		ax.set_xticklabels(['GAMMA', 'X'], minor=False, rotation=45)

		for n in range( N ):
			k =  0.5*float(n)/N*self.POSCAR.b[0,:]   +   0.5*float(n)/N*self.POSCAR.b[1,:] +  0*self.POSCAR.b[2,:] 
			data.append( np.real(np.sort(self.get_eigenvalues(k=k))) )
		
		data = np.array(data)

		plt.plot(data, 'o', ms=2)

path = '/home/akaris/Documents/code/VASP/v4.7/files/bulk_optimization/Pt/initial_files/POSCAR_FCC'
Rh = LCAO()
Rh.load(path)

Rh.BANDS()
plt.show()



