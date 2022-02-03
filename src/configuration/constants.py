import os

DATA = 'data'
RAW = 'raw'
PROCESSED = 'processed'
INTERIM = 'interim'
LOGS = 'logs'
MODELS = 'models'
REPORTS = 'reports'

# Quick access to directories.
CONFIGURATION_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
ROOT_DIRECTORY = os.path.dirname(os.path.dirname(CONFIGURATION_DIRECTORY))
DROPBOX_DIRECTORY = "/Users/lcwong/Dropbox (MIT)/hitlads"

RAW_DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, DATA, RAW)
PROCESSED_DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, DATA, PROCESSED)
INTERIM_DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, DATA, INTERIM)
LOGS_DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, DATA, LOGS)
MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, MODELS)
REPORTS_DIRECTORY = os.path.join(ROOT_DIRECTORY, REPORTS)
