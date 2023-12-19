try:
    import numpy as np
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing numpy: {str(e)}\n")
    del sys

class AtomicProperties:
    def __init__(self):

        self._plot_color = [ # pastel
            '#FFABAB',  # Salmon (Pastel)       #FFABAB    (255,171,171)
            '#A0C4FF',  # Sky Blue (Pastel)     #A0C4FF    (160,196,255)
            '#B4F8C8',  # Mint (Pastel)         #B4F8C8    (180,248,200)
            '#FFE156',  # Yellow (Pastel)       #FFE156    (255,225,86)
            '#FBE7C6',  # Peach (Pastel)        #FBE7C6    (251,231,198)
            '#AB83A1',  # Mauve (Pastel)        #AB83A1    (171,131,161)
            '#6C5B7B',  # Thistle (Pastel)      #6C5B7B    (108,91,123)
            '#FFD1DC',  # Pink (Pastel)         #FFD1DC    (255,209,220)
            '#392F5A',  # Purple (Pastel)       #392F5A    (57,47,90)
            '#FF677D',  # Watermelon (Pastel)   #FF677D    (255,103,125)
            '#FFC3A0',  # Coral (Pastel)        #FFC3A0    (255,195,160)
            '#6A057F',  # Lavender (Pastel)     #6A057F    (106,5,127)
            '#D4A5A5',  # Rose (Pastel)         #D4A5A5    (212,165,165)
            '#ACD8AA',  # Sage (Pastel)         #ACD8AA    (172,216,170)
        ]

        self._valenceElectrons = {
                "H": 1, "He": 2,
                "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "Ne": 8,
                "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7, "Ar": 8,
                "K": 1, "Ca": 2, "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11, "Zn": 12,
                "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7, "Kr": 8,
                "Rb": 1, "Sr": 2, "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Tc": 7, "Ru": 8, "Rh": 9, "Pd": 10, "Ag": 11, "Cd": 12,
                "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7, "Xe": 8,
                "Cs": 1, "Ba": 2, "La": 3, "Ce": 4, "Pr": 5, "Nd": 6, "Pm": 7, "Sm": 8, "Eu": 9, "Gd": 10, "Tb": 11, "Dy": 12, 
                "Ho": 13, "Er": 14, "Tm": 15, "Yb": 16, "Lu": 17, "Hf": 4, "Ta": 5, "W": 6, "Re": 7, "Os": 8, "Ir": 9, 
                "Pt": 10, "Au": 11, "Hg": 12, "Tl": 13, "Pb": 14, "Bi": 15, "Th": 16, "Pa": 17, "U": 18, "Np": 19, "Pu": 20
                                }

        self._atomic_mass =  {
            'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.0122, 'B': 10.81, 'C': 12.01, 'N': 14.007, 'O': 15.999, 'F': 18.998403163,
            'Ne': 20.1797, 'Na': 22.98976928, 'Mg': 24.305, 'Al': 26.9815386, 'Si': 28.085, 'P': 30.973761998, 'S': 32.06, 'Cl': 35.45,
            'Ar': 39.948, 'K': 39.0983, 'Ca': 40.078, 'Sc': 44.955908, 'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961, 'Mn': 54.938044,
            'Fe': 55.845, 'Co': 58.933194, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.723, 'Ge': 72.63, 'As': 74.921595,
            'Se': 78.971, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90584, 'Zr': 91.224, 'Nb': 92.90637,
            'Mo': 95.95, 'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.9055, 'Pd': 106.42, 'Ag': 107.8682, 'Cd': 112.414, 'In': 114.818,
            'Sn': 118.71, 'Sb': 121.76, 'Te': 127.6, 'I': 126.90447, 'Xe': 131.293, 'Cs': 132.90545196, 'Ba': 137.327, 'La': 138.90547,
            'Ce': 140.116, 'Pr': 140.90766, 'Nd': 144.242, 'Pm': 145.0, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25, 'Tb': 158.92535,
            'Dy': 162.5, 'Ho': 164.93033, 'Er': 167.259, 'Tm': 168.93422, 'Yb': 173.04, 'Lu': 174.9668, 'Hf': 178.49, 'Ta': 180.94788,
            'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217, 'Pt': 195.084, 'Au': 196.966569, 'Hg': 200.592, 'Tl': 204.38,
            'Pb': 207.2, 'Bi': 208.98040, 'Th': 232.03805, 'Pa': 231.03588, 'U': 238.05078, 'Np': 237.0, 'Pu': 244.0, 'Am': 243.0,
            'Cm': 247.0, 'Bk': 247.0, 'Cf': 251.0, 'Es': 252.0, 'Fm': 257.0, 'Md': 258.0, 'No': 259.0, 'Lr': 262.0, 'Rf': 267.0,
            'Db': 270.0, 'Sg': 271.0, 'Bh': 270.0, 'Hs': 277.0, 'Mt': 276.0, 'Ds': 281.0, 'Rg': 280.0, 'Cn': 285.0, 'Nh': 284.0,
            'Fl': 289.0, 'Mc': 288.0, 'Lv': 293.0, 'Ts': 294.0, 'Og': 294.0
                                }
        
        self._covalent_radii = {
            'H' :  .31, 'He':  .28, 'Li': 1.28, 'Be':  .96, 'B' :  .84, 'C' :  .76, 'N' :  .71, 'O' :  .66, 'F': .57,
            'Ne':  .58, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P' : 1.07, 'S' : 1.05, 'Cl': 1.02,
            'Ar': 1.06, 'K' : 1.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V' : 1.53, 'Cr': 1.39, 'Mn': 1.39,
            'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20, 'As': 1.19,
            'Se': 1.20, 'Br': 1.20, 'Kr': 1.16, 'Rb': 2.20, 'Sr': 1.95, 'Y' : 1.90, 'Zr': 1.75, 'Nb': 1.64,
            'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42,
            'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I' : 1.39, 'Xe': 1.40, 'Cs': 2.44, 'Ba': 2.15, 'La': 2.07,
            'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94,
            'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.70,
            'W' : 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45,
            'Pb': 1.46, 'Bi': 1.48, 'Th': 1.79, 'Pa': 1.63, 'U' : 1.56, 'Np': 1.55, 'Pu': 1.53, 'Am': 1.51,
            'Cm': 1.50, 'Bk': 1.50, 'Cf': 1.50, 'Es': 1.50, 'Fm': 1.50, 'Md': 1.50, 'No': 1.50, 'Lr': 1.50,
            'Rf': 1.50, 'Db': 1.50, 'Sg': 1.50, 'Bh': 1.50, 'Hs': 1.50, 'Mt': 1.50, 'Ds': 1.50, 'Rg': 1.50,
            'Cn': 1.50, 'Nh': 1.50, 'Fl': 1.50, 'Mc': 1.50, 'Lv': 1.50, 'Ts': 1.50, 'Og': 1.50
        }

        self._atomic_id = [ 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 
                            'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 
                            'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 
                            'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                            'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 
                            'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 
                            'Lv', 'Ts', 'Og']

        self._special_lattice_points = {             
                'G': np.array([0, 0, 0], dtype=np.float64),
                'L': np.array([-0.5, 0, -0.5], dtype=np.float64),
                'M': np.array([-0.5, 0.5, 0.5], dtype=np.float64),
                'N': np.array([0, 0.5, 0], dtype=np.float64),
                'R': np.array([-0.5, 0.5, 0], dtype=np.float64),
                'X': np.array([0, 0, -0.5], dtype=np.float64),
                'Y': np.array([-0.5, 0, 0], dtype=np.float64),
                'Z': np.array([0, 0.5, 0.5], dtype=np.float64),
                }

        self.element_colors = {
            "H":  (0.7, 0.7, 0.7),  # Blanco
            "He": (0.8, 0.8, 0.8),  # Gris claro
            "Li": (0.8, 0.5, 0.2),  # Marrón claro
            "Be": (0.5, 1.0, 0.0),  # Verde claro
            "B":  (0.0, 1.0, 0.0),  # Verde
            "C":  (0.2, 0.2, 0.2),  # Gris oscuro
            "N":  (0.0, 0.0, 1.0),  # Azul
            "O":  (1.0, 0.0, 0.0),  # Rojo
            "F":  (0.5, 1.0, 1.0),  # Cian claro
            "Ne": (0.8, 0.8, 1.0),  # Azul claro
            "Na": (0.0, 0.0, 0.5),  # Azul oscuro
            "Mg": (0.4, 0.8, 0.0),  # Verde oliva
            "Al": (0.8, 0.6, 0.5),  # Rosa
            "Si": (0.5, 0.5, 1.0),  # Azul medio
            "P":  (1.0, 0.5, 0.0),  # Naranja
            "S":  (1.0, 1.0, 0.0),  # Amarillo
            "Cl": (0.0, 1.0, 0.5),  # Verde menta
            "Ar": (0.5, 0.0, 0.5),  # Púrpura
            "K":  (0.6, 0.4, 0.2),  # Marrón
            "Ca": (0.3, 0.3, 0.3),  # Gris medio
            "Sc": (0.9, 0.6, 0.9),  # Lavanda
            "Ti": (0.3, 0.8, 0.8),  # Turquesa
            "V":  (0.6, 0.2, 0.2),  # Marrón rojizo
            "Cr": (0.4, 0.0, 0.0),  # Rojo oscuro
            "Mn": (0.7, 0.0, 0.7),  # Magenta
            "Fe": (0.6, 0.4, 0.0),  # Naranja oscuro
            "Co": (0.0, 0.6, 0.6),  # Verde azulado
            "Ni": (0.6, 0.6, 0.6),  # Gris plata
            "Cu": (0.7, 0.4, 0.2),  # Bronce
            "Zn": (0.5, 0.5, 0.5),   # Gris

            "Ga": (0.76, 0.56, 0.56),  # Rojo claro
            "Ge": (0.40, 0.56, 0.56),  # Verde azulado claro
            "As": (0.74, 0.50, 0.89),  # Púrpura claro
            "Se": (1.00, 0.63, 0.00),  # Naranja
            "Br": (0.65, 0.16, 0.16),  # Rojo oscuro
            "Kr": (0.36, 0.72, 0.82),  # Azul verdoso
            "Rb": (0.44, 0.18, 0.69),  # Púrpura oscuro
            "Sr": (0.00, 1.00, 0.78),  # Verde turquesa
            "Y":  (0.58, 1.00, 1.00),  # Cian
            "Zr": (0.58, 0.88, 0.88),  # Azul claro
            "Nb": (0.45, 0.76, 0.79),  # Azul verdoso
            "Mo": (0.32, 0.71, 0.71),  # Turquesa oscuro
            "Tc": (0.23, 0.62, 0.62),  # Verde azulado
            "Ru": (0.14, 0.56, 0.56),  # Verde oscuro
            "Rh": (0.04, 0.49, 0.55),  # Azul verdoso oscuro
            "Pd": (0.00, 0.41, 0.52),  # Azul marino
            "Ag": (0.75, 0.75, 0.75),  # Plata
            "Cd": (1.00, 0.85, 0.56),  # Amarillo claro
            "In": (0.65, 0.46, 0.45),  # Rojo apagado
            "Sn": (0.40, 0.50, 0.50),  # Gris azulado
            "Sb": (0.62, 0.39, 0.71),  # Lavanda
            "Te": (0.83, 0.48, 0.00),  # Naranja oscuro
            "I":  (0.58, 0.00, 0.58),  # Violeta
            "Xe": (0.26, 0.62, 0.69),  # Azul cielo
            "Cs": (0.34, 0.09, 0.56),  # Morado oscuro
            "Ba": (0.00, 0.79, 0.00),  # Verde
            "La": (0.44, 0.83, 1.00),  # Azul claro
            "Ce": (1.00, 1.00, 0.78),  # Amarillo pálido
            "Pr": (0.85, 1.00, 0.78),  # Verde claro
            "Nd": (0.78, 1.00, 0.78),  # Verde lima
            "Pm": (0.64, 1.00, 0.78),  # Verde manzana
            "Sm": (0.56, 1.00, 0.78),  # Verde agua
            "Eu": (0.38, 1.00, 0.78),  # Verde menta
            "Gd": (0.27, 1.00, 0.78),  # Turquesa claro
            "Tb": (0.19, 1.00, 0.78),  # Verde esmeralda
            "Dy": (0.12, 1.00, 0.78),  # Verde oscuro
            "Ho": (0.00, 1.00, 0.61),  # Verde mar
            "Er": (0.00, 0.90, 0.46),  # Verde bosque
            "Tm": (0.00, 0.83, 0.32),  # Verde oliva
            "Yb": (0.00, 0.75, 0.22),  # Verde musgo
            "Lu": (0.00, 0.67, 0.14),  # Verde hierba
            "Hf": (0.30, 0.76, 1.00),  # Azul celeste
            "Ta": (0.30, 0.65, 1.00),  # Azul acero
            "W":  (0.13, 0.58, 0.84),  # Azul petróleo
            "Re": (0.15, 0.49, 0.67),  # Azul denim
            "Os": (0.15, 0.40, 0.59),  # Azul marino
            "Ir": (0.09, 0.33, 0.53),  # Azul oscuro
            "Pt": (0.82, 0.82, 0.88),  # Gris perla
            "Au": (1.00, 0.82, 0.14),  # Dorado
            "Hg": (0.72, 0.72, 0.82),  # Gris metalizado
            "Tl": (0.65, 0.33, 0.30),  # Rojo cobrizo
            "Pb": (0.34, 0.35, 0.38),  # Plomo
            "Bi": (0.62, 0.31, 0.71),  # Rosa púrpura
            "Po": (0.67, 0.36, 0.00),  # Naranja cobre
            "At": (0.46, 0.31, 0.27),  # Marrón ladrillo
            "Rn": (0.26, 0.51, 0.59),  # Azul verdoso
            "Fr": (0.26, 0.00, 0.40),  # Violeta oscuro
            "Ra": (0.00, 0.49, 0.00),  # Verde oscuro
            "Ac": (0.44, 0.67, 0.98),  # Azul medio
            "Th": (0.00, 0.73, 1.00),  # Cian
            "Pa": (0.00, 0.63, 1.00),  # Azur
            "U":  (0.00, 0.56, 1.00),  # Azul ultramar
            "Np": (0.00, 0.50, 1.00),  # Azul real
            "Pu": (0.00, 0.42, 1.00),  # Azul zafiro
            "Am": (0.33, 0.36, 0.95),  # Azul lavanda
            "Cm": (0.47, 0.36, 0.89),  # Azul violeta
            "Bk": (0.54, 0.31, 0.89),  # Púrpura
            "Cf": (0.63, 0.21, 0.83),  # Magenta
            "Es": (0.70, 0.12, 0.83),  # Rosa fucsia
            "Fm": (0.70, 0.12, 0.65),  # Rosa oscuro
            "Md": (0.70, 0.05, 0.65),  # Rosa berenjena
            "No": (0.74, 0.05, 0.53),  # Rojo carmesí
            "Lr": (0.78, 0.00, 0.40),  # Rojo sangre
        }
        '''
        'X': np.array([0.5, 0, 0], dtype=np.float64),
        'W': np.array([0.5, 0.25, 0], dtype=np.float64),
        'K': np.array([0.375, 0.375, 0.75], dtype=np.float64),
        #'L': np.array([0.5, 0.5, 0.5], dtype=np.float64),
        'L': np.array([0.5, 0.0, 0.5], dtype=np.float64),
        'A': np.array([0.5, 0.5, 0.5], dtype=np.float64),
        'U': np.array([0.625, 0.25, 0], dtype=np.float64),
        
        'M': np.array([0.0, -0.5, 0.0], dtype=np.float64),
        'N': np.array([0.5, -0.5, -0.5], dtype=np.float64),
        #'R': np.array([0.0, 0.5, 0.5], dtype=np.float64),
        'R': np.array([0.5, -0.5, 0.0], dtype=np.float64),
        'Y': np.array([0.0, 0.0, 0.5], dtype=np.float64),
        'Z': np.array([0.0, -0.5, -0.5], dtype=np.float64),

        'H': np.array([0.5, -0.5, 0.5], dtype=np.float64),
        #'N': np.array([0, 0, 0.5], dtype=np.float64),
        'P': np.array([0.25, 0.25, 0.25], dtype=np.float64),
        'B': np.array([0, 0.25, 0], dtype=np.float64),
        'C': np.array([0, 0, 0.25], dtype=np.float64),
        'D': np.array([0.25, 0.25, 0], dtype=np.float64),
        'E': np.array([0.25, 0, 0.25], dtype=np.float64),
        'F': np.array([0, 0.25, 0.25], dtype=np.float64),
        'Gp': np.array([0.25, 0.25, 0.25], dtype=np.float64),
        'H': np.array([0.5, 0, 0], dtype=np.float64),
        'I': np.array([0, 0.5, 0], dtype=np.float64),
        'J': np.array([0, 0, 0.5], dtype=np.float64),  
        '''
    