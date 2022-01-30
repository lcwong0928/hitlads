import glob
import os
import pickle
from dataclasses import dataclass

import pandas as pd

from src.configuration.constants import INTERIM_DATA_DIRECTORY


@dataclass
class InterimDataset:
    # Metadata
    source: str
    dataset: str
    signal: str

    # Train
    X_train: pd.DataFrame
    index_train: pd.Series
    labels_train: pd.Series
    anomalies_train: pd.DataFrame

    # Test
    X_test: pd.DataFrame
    index_test: pd.Series
    labels_test: pd.Series
    anomalies_test: pd.DataFrame

    @classmethod
    def load(cls, source: str, dataset: str, signal: str):
        data_directory = os.path.join(INTERIM_DATA_DIRECTORY, source, dataset, signal, 'dataset.pickle')
        with open(data_directory, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    @classmethod
    def list_signals(cls, source: str, dataset: str) -> list:
        signals = []
        for filepath in glob.glob(os.path.join(INTERIM_DATA_DIRECTORY, source, dataset, '**')):
            signals.append(os.path.basename(filepath))
        return signals

    def save(self) -> None:
        output_directory = os.path.join(INTERIM_DATA_DIRECTORY, self.source, self.dataset, self.signal)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        with open(os.path.join(output_directory, 'dataset.pickle'), 'wb') as f:
            pickle.dump(self, f)
