import numpy as np 	
import matplotlib.pyplot as plt

class INCAR(object):
	def __init__(self, file_name=None, plot_ions=None, plot_orbitals=None):
		self.file_name = file_name

		self.defaut = {
# ==== INICILIZATION ==== #
'ISTART' 			:	'0', 
'ICHARG' 			:	'2',		# ICHARG = 0 | 1 | 2 | 4  
'ISPIN' 			:	'1', 		# ISPIN = 1 | 2
'ISMEAR' 			:	'1', 		# ISMEAR = -5 | -4 | -3 | -2 | -1 | 0 | [integer]>0

# ==== KSPACING ==== #
'KSPACING' 			:	'0.5', 		# KSPACING = [real]
'KGAMMA' 			:	'True', 	# KGAMMA = [logical]

# ==== electronic optimization ==== #
'ENCUT' 			:	'450',  	# Default: ENCUT 	= largest ENMAX on the POTCAR file 	
'NELM' 				:	'60', 		# NELM = [integer]
'ALGO' 				:	'Normal', 	# ALGO = Normal | VeryFast | Fast | Conjugate | All | Damped | Subrot | Eigenval | Exact | None |
									# Nothing | CHI | G0W0 | GW0 | GW | scGW0 | scGW | G0W0R | GW0R | GWR | scGW0R | scGWR | ACFDT | RPA | 
									# ACFDTR | RPAR | BSE | TDHF
'EDIFF' 			:	'0.0001', 	# NELM = [integer]
'SIGMA' 			:	'0.2', 		# SIGMA = [real]

# ==== ionic relaxation ==== #
'EDIFFG' 			:	'0.001', 	# Default: EDIFFG = EDIFF×10  
'NSW' 				:	'0', 		# Default: NSW = 0  
'IBRION' 			:	'0', 		# Default: IBRION 	= -1 	for NSW=−1 or 0  ||  IBRION = -1 | 0 | 1 | 2 | 3 | 5 | 6 | 7 | 8 | 44  
'POTIM' 			:	'0.5', 		# POTIM 	= none, 	must be set if IBRION= 0 (MD) ||  0.5 	if IBRION= 1, 2, and 3 (ionic relaxation), and 5 (up to VASP.4.6)  || 0.015 	if IBRION=5, and 6 (as of VASP.5.1) 
'ISIF' 				:	'0', 		# ISIF = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7   Default: ISIF 	= 0 	for IBRION=0 (molecular dynamics) or LHFCALC=.TRUE. 

# ==== performance optimization ==== #
'PREC' 				:	'Normal',	# PREC = Low | Medium | High | Normal | Single | Accurate  
'KPAR' 				:	'1', 		# KPAR = [integer]
'NPAR' 				:	'1', 		# Default: NPAR = number of cores  
'NCORE' 			:	'1', 		# NCORE = [integer]

# ==== VDW ==== #
'IVDW' 				:	'0', 		# IVDW = 0 | 1 | 10 | 11 | 12 | 2 | 20 | 21 | 202 | 4
		
# ==== extraFILES ==== # 
'LORBIT' 			:	'None', 		# LORBIT = 0 | 1 | 2 | 5 | 10 | 11 | 12  
	
		}

# "Here is a short overview of all parameters currently supported. Parameters which are used frequently are emphasized.
		self.help = {
'NGX'				:	'FFT mesh for orbitals (Sec. 6.3,6.11)',
'NGY'				: 	'FFT mesh for orbitals (Sec. 6.3,6.11)',
'NGZ'				: 	'FFT mesh for orbitals (Sec. 6.3,6.11)',
'NGXF'				:	'FFT mesh for charges (Sec. 6.3,6.11)',
'NGYF'				:	'FFT mesh for charges (Sec. 6.3,6.11)',
'NGZF'				: 	'FFT mesh for charges (Sec. 6.3,6.11)',
'NBANDS'			: 	'number of bands included in the calculation (Sec. 6.5)',
'NBLK'				: 	'blocking for some BLAS calls (Sec. 6.6)',
'INIWAV'			: 	'initial electr wf. : 0-lowe 1-rand',


'NELMIN'			: 	'nr. of electronic steps',
'NELMDL'			: 	'nr. of electronic steps',

'NBLOCK'			:	'number of steps for ionic upd.',
'KBLOCK'			:	'inner block; outer block',

'ISIF'				: 	'calculate stress and what to relax',
'IWAVPR'			: 	'prediction of wf.: 0-non 1-charg 2-wave 3-comb',
'ISYM' 				:	'symmetry: 0-nonsym 1-usesym',
'SYMPREC'			: 	'precession in symmetry routines',
'LCORR'				: 	'Harris-correction to forces',
'TEBEG'				:  	'temperature during run',
'TEEND'				: 	'temperature during run',
'SMASS'				: 	'Nose mass-parameter (am)',
'NPACO'				: 	'distance and nr. of slots for P.C.',
'APACO'				:	'distance and nr. of slots for P.C.',
'POMASS'			: 	'mass of ions in am',
'ZVAL'				: 	'ionic valence',
'RWIGS'				: 	'Wigner-Seitz radii',
'NELECT'			: 	'total number of electrons',
'NUPDOWN'			: 	'fix spin moment to specified value',
'EMIN'				:	'energy-range for DOSCAR file',
'EMAX'				: 	'energy-range for DOSCAR file',

'IALGO'				: 	'algorithm: use only 8 (CG) or 48 (RMM-DIIS)',
'GGA'				: 	'xc-type: e.g. PE AM or 91',
'VOSKOWN'			: 	'use Vosko, Wilk, Nusair interpolation',
'DIPOL'				:	'center of cell for dipol',
'AMIX'				:	'tags for mixing',
'BMIX'				: 	'tags for mixing',
'WEIMIN'			:	'special control tags',
'EBREAK'			: 	'special control tags',
'DEPER'				: 	'special control tags',
'TIME'				: 	'special control tag',
'LWAVE'				:	'create WAVECAR/CHGCAR/LOCPOT',
'LCHARG'			:	'create WAVECAR/CHGCAR/LOCPOT',
'LVTOT'				:	'create WAVECAR/CHGCAR/LOCPOT',
'LVHAR'				: 	'create WAVECAR/CHGCAR/LOCPOT',
'LELF'				: 	'create ELFCAR',
'LORBIT'			: 	'LORBIT, together with an appropriate RWIGS, determines whether the PROCAR or PROOUT files are written.',
'LSCALAPACK' 		:	'switch off scaLAPACK',
'LSCALU'			: 	'switch of LU decomposition',
'LASYNC'			: 	'overlap communcation with calculations',

'SYSTEM'			: 	'name of System',
'NWRITE'			: 	'verbosity write-flag (how much is written)',

# ==== start parameters for this Run (automatic defaults, hence not often required) ==== #
# >> initialization
'ICHARG' 			:	'charge: 1-file 2-atom 10-const. Default: ICHARG = 2 if ISTART=0, = 0 else. charge: 1-file 2-atom 10-const',
'ISTART' 			:	'job : 0-new 1-cont 2-samecut. Default: ISTART = 1 if WAVECAR exists, = 0 else ISTART 	meaning \n 0 	Calculate charge density from initial orbitals. \n 2 	Take superposition of atomic charge densities \n 11 	To obtain the eigenvalues (for band structure plots) or the DOS for a given charge density read from CHGCAR. The selfconsistent CHGCAR file must be determined beforehand doing by a fully selfconsistent calculation with a k-point grid spanning the entire Brillouin zone.\n 12 	Non-selfconsistent calculations for a superposition of atomic charge densities.',
'ISMEAR'			:	'Description: ISMEAR determines how the partial occupancies fnk are set for each orbital. part. occupancies: -5 Blöchl -4-tet -1-fermi 0-gaus >0 MP',

# >> spin definition
'ISPIN' 			:	'spin polarized calculation (2-yes 1-no)',
'MAGMOM'			: 	'initial mag moment / atom (NIONS * 1.0 for ISPIN=2 )',

# ==== electronic optimization ==== #
'ENCUT' 			:	'ENCUT specifies the cutoff energy for the plane-wave-basis set in eV.', # ENCUT 	= largest ENMAX on the POTCAR file 	
'ALGO'				: 	'algorithm: Normal (Davidson) | Fast | Very_Fast (RMM-DIIS). Description: the ALGO tag is a convenient option to specify the electronic minimization algorithm (as of VASP.4.5) and/or to select the type of GW calculations',
'NELM'				: 	'NELM sets the maximum number of electronic SC (self-consistency) steps.',
'EDIFF'				: 	'stopping-criterion for electronic upd. EDIFF specifies the global break condition for the electronic SC-loop. EDIFF is specified in units of eV',
'SIGMA'				:   'broadening in eV -4-tet -1-fermi 0-gaus. Description: SIGMA specifies the width of the smearing in eV.',

# ==== ionic relaxation ==== #
'EDIFFG'			:	'EDIFFG defines the break condition for the ionic relaxation loop.',
'NSW'				:	'NSW sets the maximum number of ionic steps.',
'IBRION'			: 	'ionic relaxation: 0-MD 1-quasi-New 2-CG. Description: IBRION determines how the ions are updated and moved. ',
'POTIM'				:	'time-step for ion-motion (fs). POTIM sets the time step in molecular dynamics or the step width in ionic relaxations',

# ==== KSPACING ==== #
'KSPACING'			: 	'Description: The tag KSPACING determines the number of k points if the KPOINTS file is not present.',
'KGAMMA'			: 	'Determines whether the k points (specified by the KSPACING tag ) include (KGAMMA=.TRUE.) the Gamma point',

# ==== performance optimization #
# For PREC=Medium and PREC=Accurate, ENCUT will be set to maximal ENMAX value found on the POTCAR file
'PREC' 				:	'precession: medium, high or low : VASP.4.5 also: normal, accurate. The PREC-flag determines the energy cutoff ENCUT, if (and only if) no value is given for ENCUT in the INCAR file. For PREC=Low, ENCUT will be set to the maximal ENMIN value found in the POTCAR files. For PREC=Medium and PREC=Accurate, ENCUT will be set to maximal ENMAX value found on the POTCAR file.',
'ENCUT' 			:	'Description: ENCUT specifies the cutoff energy for the plane-wave-basis set in eV.',
'ROPT'				: 	'Description: ROPT determines how accurately the projectors are represented in real space. number of grid points for non-local proj in real space',
'KPAR'				: 	'KPAR determines the number of k-points that are to be treated in parallel (available as of VASP.5.3.2). Also, KPAR is used as parallelization tag for Laplace transformed MP2 calculations.',
'NCORE'				: 	'NCORE determines the number of compute cores that work on an individual orbital (available as of VASP.5.2.13). ',
'LREAL'				: 	'LREAL determines whether the projection operators are evaluated in real-space or in reciprocal space.',
'NPAR'				: 	'Parallelization over bands, On massively parallel systems and modern multi-core machines we strongly urge to set ',

# ==== VDW ==== #
'IVDW'				:	'IVDW specifies a vdW (dispersion) correction',
'LAECHG'			:	'???',
'LUSE_VDW'			:	'???',
'AGGAC'				:	'???',
'LASPH'				:	'???',
'PARAM1'			:	'???',
'PARAM2'			:	'???',
'Zab_vdW'			:	'???',
'BPARAM'			:	'???',
'METAGGA'			:	'???',
''					:	'???',
	}
		'''
		# -------------------------------------------------------------------------- #
		# The GGA tag is further used to choose the appropriate exchange functional. #

		# -- original vdW-DF of Dion et al uses revPBE -- # 
		GGA = RE
		LUSE_VDW = .TRUE.
		AGGAC = 0.0000
		LASPH = .TRUE.

		# --  For optPBE-vdW set -- #
		#GGA = OR
		#LUSE_VDW = .TRUE.
		#AGGAC = 0.0000
		#LASPH = .TRUE.

		# -- optB88-vdW set: -- # 
		#GGA = BO
		#PARAM1 = 0.1833333333
		#PARAM2 = 0.2200000000
		#LUSE_VDW = .TRUE.
		#AGGAC = 0.0000
		#LASPH = .TRUE.

		# -- optB86b-vdW: -- #
		#GGA = MK 
		#PARAM1 = 0.1234 
		#PARAM2 = 1.0000
		#LUSE_VDW = .TRUE.
		#AGGAC = 0.0000
		#LASPH = .TRUE.

		# -- vdW-DF2, set: -- #
		#GGA = ML
		#LUSE_VDW = .TRUE.
		#Zab_vdW = -1.8867
		#AGGAC = 0.0000
		#LASPH = .TRUE.

		# -- The rev-vdW-DF2 functional of Hamada, 
		# also known as vdW-DF2-B86R -- #
		#GGA      = MK
		#LUSE_VDW = .TRUE.
		#PARAM1   = 0.1234
		#PARAM2   = 0.711357
		#Zab_vdW  = -1.8867
		#AGGAC    = 0.0000
		#LASPH = .TRUE.

		# -- the SCAN + rVV10 functional set -- #
		#METAGGA  = SCAN
		#LUSE_VDW = .TRUE.
		#BPARAM = 15.7
		#LASPH = .TRUE.
		# it is NOT  possible to combine SCAN with vdW-DFT functionals other than rVV10.


		# Bader
		#LAECHG = T
		'''

		self.attr_dic = {}

	def isnum(self, n):
		# ------------------ Define if n is or not a number ------------------ # 
		# n     :   VAR     :   VAR to check if it is a numerical VAR
		# return :  BOOL    : True/False
		try: float(n); return True
		except: return False

	def var_assing(self, var_name, var_value):
		if not var_name in self.attr_dic.keys():
			self.attr_dic[var_name] = var_value
			setattr(self, var_name, var_value)
		else:
			self.attr_dic[var_name] = var_value
			setattr(self, var_name, var_value)
			
	def load(self, file_name=None):
		if file_name is None: file_name = self.file_name

		with open(file_name,'r') as f:

			for i, n in enumerate(f):
				try:
					vec = [m for m in n.split(' ') if m != '' and m != '\n']
					if len(vec) > 2 and not '#' in vec[0]:	
						if self.isnum(vec[2]): var_name, var_value = vec[0], float(vec[2])
						else: var_name, var_value = vec[0], vec[2].split('\n')[0] 
						self.var_assing(var_name, var_value)
					if len(vec) > 1 and not '#' in vec[0] and '=' in vec[0]: 
						if self.isnum(vec[1]): var_name, var_value = vec[0], float(vec[1])
						else: var_name, var_value = vec[0].split('=')[0], vec[1].split('\n')[0] 
						self.var_assing(var_name, var_value)
				except: print('can not read line {} :: '.format(i+1), n )

	def isnum(self, n):
		# ------------------ Define if n is or not a number ------------------ # 
		# n     :   VAR     :   VAR to check if it is a numerical VAR
		# return :  BOOL    : True/False
		try: float(n); return True
		except: return False


	def isINT(self, num): return self.isnum(num) and abs(num - int(num)) < 0.0001 

	def view(self, ):
		for n in self.attr_dic.keys():
			if n in self.help.keys():	info = self.help[n]
			else:						info = 'unknow'
			if self.isINT(self.attr_dic[n]):
				print( f'{n:<10.10s} : {int(self.attr_dic[n]):<10} : {info:<80.80s}[...]' )
			else:
				print( f'{n:<10.10s} : {self.attr_dic[n]:<10.10} : {info:<80.80s}[...]' )

	def summary(self, ):
		self.view()

	def save(self, file:str=None):
		if file is None: file = self.file_name

		with open(file, 'w') as f:
			for n in self.attr_dic.keys():
				if n in self.help.keys():	info = self.help[n]
				else:						info = 'unknow'
				if self.isINT(self.attr_dic[n]):
					f.write( f'{n:<15.15s} = {int(self.attr_dic[n]):<20} ! {info:<80.80s}[...]\n' )
				else:
					f.write( f'{n:<15.15s} = {self.attr_dic[n]:<20.20} ! {info:<80.80s}[...]\n' )

if __name__ == "__main__":
	# How to...		
	incar = INCAR()
	incar.load('/home/akaris/Documents/code/VASP/v4.7/files/INCAR/INCAR')
	incar.view()
	#print(IN.view())






