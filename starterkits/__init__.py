import logging as _logging
from logging.handlers import RotatingFileHandler as _RotatingFileHandler

import os as _os

from pathlib import Path
 
PROJECT_PATH = Path(__file__).parent.parent.absolute()
DATA_PATH = PROJECT_PATH / 'data' 
LOG_PATH = PROJECT_PATH / 'log'

if not _os.path.exists(LOG_PATH):
    _os.makedirs(LOG_PATH)

_logging.basicConfig(
    level=_logging.INFO,
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
    handlers=[
        _RotatingFileHandler(
            "{0}/{1}.log".format(LOG_PATH, "log"),
            maxBytes=(1048576*5), backupCount=7),
        _logging.StreamHandler()
    ])
