# ©, 2022, Sirris
# owner: FFNG
"""
All relevant paths are stored in the __init__.py file:
"""

import os
import posixpath
from pathlib import Path
import pkgutil as _pkgutil

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_PATH = os.path.join('..', '..', 'data')
FNAME = 'Data_Challenge_PHM2023_training_data'

BASE_PATH_HEALTHY = os.path.join(DATA_PATH, FNAME, 'Pitting_degradation_level_0 (Healthy)')

PITTING_LEVELS = [1, 2, 3, 4, 6, 8]
BASE_PATHS_TEST = [os.path.join(DATA_PATH, FNAME, f'Pitting_degradation_level_{pl}') for pl in PITTING_LEVELS]

SETUP = {'start': 0.5, 'stop': 100.5, 'n_windows': 50, 'window_steps': 2, 'window_size': 2}
