"""
Put configs into global variables.
"""

import multiprocessing as mp
import numpy as np
from helper.io import json2dict, csv2list

IPOPT_SETTING = {
    'ipopt': {
        'print_level': 0,
        'sb': 'yes',
        'acceptable_tol': 1e-12,
    },
    'print_time': 0,
}

QPOASES_SETTING = {
    'terminationTolerance': 1e-16,
    'boundTolerance': 1.0e-16,
}

SYMBOL_DEBUG = True

CONFIGS: dict = json2dict('config.json')

ALADIN_CFGS = dict()
for acn in ['max_iter', 'tol', 'copied_gap']:
    ALADIN_CFGS[acn.upper()] = CONFIGS['aladin']['config'][acn]

SAMPLE_N1: int = CONFIGS['sampling']['N1']
SAMPLE_N2: int = CONFIGS['sampling']['N2']

POOL_CNT: int = mp.cpu_count() // CONFIGS['performance']['cpu_div']

V_INITS: list = csv2list('data/cars.csv')
T_GUESS: list = csv2list('data/t_guess.csv')

# UNIT CONVERSION: km/h -> m/s
def kmph2mps(d: dict) -> dict:
    d['V0'] /= 3.6
    d['Vref'] /= 3.6
    return d

# Apply unit conversion
V_INITS = [kmph2mps(d) for d in V_INITS]
# V_INITS = list(map(kmph2mps, V_INITS))

SUB_SYS_COUNT: int = len(V_INITS)

PRINT_LEVEL: int = CONFIGS['debug']['print_level']
