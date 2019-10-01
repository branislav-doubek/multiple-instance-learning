import logging

logging.basicConfig(level=logging.ERROR,
                    format='%(process)d-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)