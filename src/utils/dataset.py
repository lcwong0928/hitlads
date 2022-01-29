import os
import pickle

from src.configuration.constants import INTERIM_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY, RAW_DATA_DIRECTORY, INTERIM, \
    PROCESSED, RAW

DIRNAME_TO_DIRECTORY = {
    INTERIM: INTERIM_DATA_DIRECTORY,
    PROCESSED: PROCESSED_DATA_DIRECTORY,
    RAW: RAW_DATA_DIRECTORY,
}


def load_dataset(dirname: str, filename: str):
    directory = DIRNAME_TO_DIRECTORY[dirname]
    pickle_extension = '.pickle'
    if pickle_extension not in filename:
        filename += pickle_extension

    filepath = os.path.join(directory, filename)

    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)

    return dataset
