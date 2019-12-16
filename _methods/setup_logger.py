import logging

#logging.basicConfig(level=logging.DEBUG,
#                    format='%(levelname)s - %(message)s')
#logger = logging.getLogger(__name__)


import logging
import os

RED = '\033[1;31m'
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[1;34m'
WHITE = '\033[1;37m'

RESET = '\033[0m'

FORMAT = f'[{GREEN}%(name)s{RESET}][%(levelname)s] %(message)s'

COLORS = {
    'WARNING': YELLOW,
    'INFO': RED,
    'DEBUG': BLUE,
    'ERROR': RED
}

LEVELS = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
}

_LOGLEVEL_NAME = os.environ.get('LOGLEVEL', 'INFO').upper()
LOGLEVEL = LEVELS.get(_LOGLEVEL_NAME, logging.INFO)

class ColoredFormatter(logging.Formatter):
    def init(self, fmt):
        super().init(fmt)

    def format(self, record):
        levelname = record.levelname
        if levelname in COLORS:
            color = COLORS[levelname]
            levelname_color = f'{color}{levelname}{RESET}'
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)

def get_logger(name):
    logger = logging.getLogger(name)
    formatter = ColoredFormatter(FORMAT)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.setLevel(logging.ERROR)
    logger.addHandler(sh)

    return logger

logger = get_logger(__name__)
