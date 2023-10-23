try:
    FileManager = __import__('FileManager').FileManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing FileManager: {str(e)}\n")
    del sys

try:
    KPointsManager = __import__('KPointsManager').KPointsManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing KPointsManager: {str(e)}\n")
    del sys

try:
    InputDFT = __import__('InputDFT').InputDFT
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing InputDFT: {str(e)}\n")
    del sys

try:
    PotentialManager = __import__('PotentialManager').PotentialManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing PotentialManager: {str(e)}\n")
    del sys

try:
    PeriodicSystem = __import__('PeriodicSystem').PeriodicSystem
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing PeriodicSystem: {str(e)}\n")
    del sys

try:
    import numpy as np
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing numpy: {str(e)}\n")
    del sys

class OutFileManager(FileManager):
    def __init__(self, file_location:str=None, name:str=None, **kwargs):
        """
        Initialize OutFileManager class.
        :param file_location: Location of the file to be read.
        :param name: Name identifier for the file.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, file_location=file_location)
        self._comment = None
        self._KPointsManager = None
        self._AtomPositionManager = None
        self._PotentialManager = None
        self._InputFileManager = None

        self._total_charge = None

        self._parameters = None
        self.parameters_data = ['SYSTEM', 'POSCAR', 'Startparameter for this run:', 'NWRITE', 'PREC', 'ISTART', 
                                'ICHARG', 'ISPIN', 'LNONCOLLINEAR', 'LSORBIT', 'INIWAV', 'LASPH', 'METAGGA', 
                                'Electronic Relaxation 1', 'ENCUT', 'ENINI', 'ENAUG', 'NELM', 'EDIFF', 'LREAL', 
                                'NLSPLINE', 'LCOMPAT', 'GGA_COMPAT', 'LMAXPAW', 'LMAXMIX', 'VOSKOWN', 'ROPT', 
                                'ROPT', 'Ionic relaxation', 'EDIFFG', 'NSW', 'NBLOCK', 'IBRION', 'NFREE', 'ISIF',
                                'IWAVPR', 'ISYM', 'LCORR', 'POTIM', 'TEIN', 'TEBEG', 'SMASS', 'estimated Nose-frequenzy (Omega)', 
                                'SCALEE', 'NPACO', 'PSTRESS', 'Mass of Ions in am', 'POMASS', 'Ionic Valenz', 
                                'ZVAL', 'Atomic Wigner-Seitz radii', 'RWIGS', 'virtual crystal weights', 'VCA', 
                                'NELECT', 'NUPDOWN', 'DOS related values:', 'EMIN', 'EFERMI', 'ISMEAR', 
                                'Electronic relaxation 2 (details)', 'IALGO', 'LDIAG', 'LSUBROT', 'TURBO', 
                                'IRESTART', 'NREBOOT', 'NMIN', 'EREF', 'IMIX', 'AMIX', 'AMIX_MAG', 'AMIN', 'WC', 
                                'Intra band minimization:', 'WEIMIN', 'EBREAK', 'DEPER', 'TIME', 'volume/ion in A,a.u.', 
                                'Fermi-wavevector in a.u.,A,eV,Ry', 'Thomas-Fermi vector in A', 'Write flags', 
                                'LWAVE', 'LCHARG', 'LVTOT', 'LVHAR', 'LELF', 'LORBIT', 'Dipole corrections', 
                                'LMONO', 'LDIPOL', 'IDIPOL', 'EPSILON', 'Exchange correlation treatment:', 'GGA', 
                                'LEXCH', 'VOSKOWN', 'LHFCALC', 'LHFONE', 'AEXX', 'Linear response parameters', 
                                'LEPSILON', 'LRPA', 'LNABLA', 'LVEL', 'LINTERFAST', 'KINTER', 'CSHIFT', 'OMEGAMAX', 
                                'DEG_THRESHOLD', 'RTIME', 'Orbital magnetization related:', 'ORBITALMAG', 'LCHIMAG', 'DQ']

    def readOUTCAR(self, file_location:str=None):
        """
        Read the OUTCAR file and parse its contents.
        :param file_location: Optional file location if different from self.file_location.
        """
        file_location = file_location if type(file_location) == str else self.file_location
        lines =list(self.read_file(file_location,strip=False))
        
        read_parameters = True
        parameters = {}
        
        self._InputFileManager = InputDFT()
        self._KPointsManager = KPointsManager()
        self._AtomPositionManager = []
        self._PotentialManager = PotentialManager()

        APM = PeriodicSystem()

        CL, TC, MG, TF, IR = -1, 0, 0, 0, 0 # mark for 

        for i, line in enumerate(lines): # read the file {file_name} line by line
            line_vec = [x for x in line.strip().split(' ') if x]

            if read_parameters:
                if 'position of ions in fractional' in line:
                    read_parameters = False
                else:
                    var_name = line.strip().split('=')[0].strip()
                    if var_name in self.parameters_data:
                        parameters[var_name] = ' '.join( line.strip().split('=')[1:] )

                    if var_name in self._InputFileManager.parameters_data:
                        self._InputFileManager.parameters[var_name] = ' '.join( line.strip().split('=')[1:] )

            elif 'E-fermi' in line : 
                self.E_fermi = line_vec[2]

            if 'NIONS' in line: 
                self.NIONS = int(line_vec[11])
                APM._atomCount = int(line_vec[11])

            # === POTCAR === #
            if 'POTCAR' in line: 
                try:    
                    if not line.split(':')[1] in self.POTCAR_full:
                        self.POTCAR_full.append(line.split(':')[1])

                    if not line.split(':')[1].strip().split(' ')[1] in self.POTCAR:
                        self.POTCAR.append( line.split(':')[1].strip().split(' ')[1] )

                except: pass

            # --- store CELL --- # dim = steps, 3, 3
            if 'direct lattice vectors' in line: 
                CL = 0
                cell = []
                continue
            if int(CL) >= 0:
                CL += 1
                if int(CL) < 4:
                    cell.append( [float(line_vec[0]),float(line_vec[1]),float(line_vec[2])] )
                else: 
                    CL = -1
                    APM.cell = np.array(cell)

            # --- store CHAGE --- # dim = steps, ions, 4
            if 'total charge' in line and not 'charge-density' in line: 
                TC = self.NIONS+4
                total_charge = np.zeros((self.NIONS,4))
                TC_counter = 0 

            if int(TC) > 0: 
                TC -= 1
                if int(TC) < int(self.NIONS):
                    total_charge[TC_counter,:] = [float(line_vec[1]),float(line_vec[2]),float(line_vec[3]),float(line_vec[4])] 
                    TC_counter += 1 
                if TC == 0: 
                    APM.total_charge = np.array(total_charge)


            # --- store CHAGE --- # dim = steps, ions, 4
            if 'magnetization (x)' in line: MG = self.NIONS+4; magnetization = []
            if int(MG) > 0 :
                MG -= 1
                if int(MG) < self.NIONS:
                    magnetization.append( [float(line_vec[1]),float(line_vec[2]),float(line_vec[3]),float(line_vec[4])] )
                if MG == 0: 
                    APM._magnetization.append( np.array(magnetization))
            
            # --- store IR displacement --- # dim = steps, ions, 4
            if '2PiTHz' in line: IR = int(self.NIONS)+2; IRdisplacement = []; self.vibrations.append([float(line_vec[3]),float(line_vec[5]),float(line_vec[7]),float(line_vec[9])])
            if int(IR) > 0 :
                IR -= 1
                if int(IR) < int(self.NIONS):
                    IRdisplacement.append( [   float(line_vec[0]),float(line_vec[1]),float(line_vec[2]),
                                               float(line_vec[3]),float(line_vec[4]),float(line_vec[5])] )
                if IR == 0: 
                    self.IRdisplacement.append( np.array(IRdisplacement))


            # --- store CHAGE --- # dim = steps, 1
            if 'Edisp' in line: 
                APM._Edisp = line_vec[-1][:-1]

            # --- store FORCE --- # dim = steps-1, ions, 6
            if 'TOTAL-FORCE' in line: 
                TF = int(self.NIONS)+2
                TF_counter = 0
                total_force = []

            if int(TF) > 0 :
                TF -= 1
                if int(TF) < int(self.NIONS):
                    total_force = [float(line_vec[3]),float(line_vec[4]),float(line_vec[5])] 
                    positions   = [float(line_vec[0]),float(line_vec[1]),float(line_vec[2])] 
                    TF_counter += 1
                if TF == 0: 
                    APM._total_force = np.array(total_force)

            if 'energy  without entropy=' in line:
                APM._E = float(line_vec[-1]) 
                self._AtomPositionManager.append(APM)

'''
o = OutFileManager('/home/akaris/Documents/code/Physics/VASP/v6.1/files/OUTCAR/Sys/yuwa/LLZO_OUTCAR')
o.readOUTCAR()
print( [ APM.E for APM in self._AtomPositionManager ])
'''
