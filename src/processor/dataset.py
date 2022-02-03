import glob
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.configuration.constants import PROCESSED_DATA_DIRECTORY


@dataclass
class Dataset:
    name: str
    data: pd.DataFrame

    index: np.array = np.array([])
    X: np.ndarray = np.array([])
    y: np.array = np.array([])
    labels: np.array = np.array([])

    attributes: tuple = ('index', 'X', 'y', 'labels')

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save(self, directory: str) -> None:
        output_directory = os.path.join(directory, self.name)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        self.data.to_parquet(os.path.join(output_directory, 'data.parquet'))

        for attribute in self.attributes:
            np.save(os.path.join(output_directory, f'{attribute}.npy'), self.__getattribute__(attribute))

    @classmethod
    def load(cls, directory: str):
        name = os.path.basename(directory)
        data = pd.read_parquet(os.path.join(directory, 'data.parquet'))
        dataset = Dataset(name, data)

        for attribute in cls.attributes:
            dataset.__setattr__(attribute, np.load(os.path.join(directory, f'{attribute}.npy')))
        return dataset


@dataclass
class AnomalyDataset:
    source: str
    dataset: str
    signal: str

    train: Dataset
    test: Dataset
    anomalies: pd.DataFrame

    def save(self, directory: str = PROCESSED_DATA_DIRECTORY) -> None:
        output_directory = os.path.join(directory, self.source, self.dataset, self.signal)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        self.train.save(output_directory)
        self.test.save(output_directory)
        self.anomalies.to_parquet(os.path.join(output_directory, 'anomalies.parquet'))

    @classmethod
    def load(cls, source: str, dataset: str, signal: str, directory: str = PROCESSED_DATA_DIRECTORY):
        directory = os.path.join(directory, source, dataset, signal)
        train = Dataset.load(os.path.join(directory, 'train'))
        test = Dataset.load(os.path.join(directory, 'test'))
        anomalies = pd.read_parquet(os.path.join(directory, 'anomalies.parquet'))
        dataset = AnomalyDataset(source, dataset, signal, train, test, anomalies)
        return dataset

    @classmethod
    def get_sources(cls, directory: str = PROCESSED_DATA_DIRECTORY) -> list:
        directory = os.path.join(directory, '**')
        return [os.path.basename(filepath) for filepath in glob.glob(directory)]

    @classmethod
    def get_datasets(cls, source: str, directory: str = PROCESSED_DATA_DIRECTORY) -> list:
        directory = os.path.join(directory, source, '**')
        return [os.path.basename(filepath) for filepath in glob.glob(directory)]

    @classmethod
    def get_signals(cls, source: str, dataset: str, directory: str = PROCESSED_DATA_DIRECTORY) -> list:
        directory = os.path.join(directory, source, dataset, '**')
        return [os.path.basename(filepath) for filepath in glob.glob(directory)]
