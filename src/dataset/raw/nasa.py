import os

import numpy as np
import pandas as pd

from src.configuration.constants import RAW_DATA_DIRECTORY
from src.dataset.interim_dataset import InterimDataset


class NASADataset:

    @classmethod
    def raw_to_interim(cls):
        anomalies_df = pd.read_csv(os.path.join(RAW_DATA_DIRECTORY, 'NASA/labeled_anomalies.csv'))
        source = 'NASA'

        for idx, row in anomalies_df.iterrows():
            dataset = row['spacecraft']
            signal = row['chan_id']
            anomalies = eval(row['anomaly_sequences'])

            X_train = pd.DataFrame(np.load(os.path.join(RAW_DATA_DIRECTORY, f'NASA/train/{signal}.npy')))
            index_train = X_train.reset_index()['index']
            anomalies_train = pd.DataFrame(columns=['start', 'end'])
            labels_train = cls.anomalies_to_labels(index_train, anomalies_train)

            X_test = pd.DataFrame(np.load(os.path.join(RAW_DATA_DIRECTORY, f'NASA/test/{signal}.npy')))
            index_test = X_test.reset_index()['index']
            anomalies_test = pd.DataFrame(anomalies, columns=['start', 'end'])
            labels_test = cls.anomalies_to_labels(index_test, anomalies_test)

            InterimDataset(source, dataset, signal,
                           X_train, index_train, labels_train, anomalies_train,
                           X_test, index_test, labels_test, anomalies_test).save()

    @classmethod
    def anomalies_to_labels(cls, index: pd.DataFrame, anomalies: pd.DataFrame) -> pd.Series:
        index = index.copy()
        labels = [0] * len(index)
        for start, end in zip(anomalies.start, anomalies.end):
            for i in range(start, end + 1):
                labels[i] = 1
        return pd.Series(labels, name='labels')
