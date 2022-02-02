import os

import numpy as np
import pandas as pd

from src.configuration.constants import RAW_DATA_DIRECTORY, INTERIM_DATA_DIRECTORY
from src.processor.dataset import AnomalyDataset, Dataset


class NASAProcessor:
    source = 'NASA'

    @classmethod
    def process(cls, input_directory: str = RAW_DATA_DIRECTORY, out_directory: str = INTERIM_DATA_DIRECTORY):
        anomalies_df = pd.read_csv(os.path.join(input_directory, 'NASA/labeled_anomalies.csv'))

        for idx, row in anomalies_df.iterrows():
            dataset = row['spacecraft']
            signal = row['chan_id']
            anomalies = pd.DataFrame(eval(row['anomaly_sequences']), columns=['start', 'end'])

            filepath = os.path.join(RAW_DATA_DIRECTORY, 'NASA/{}/{}.npy')

            X_train = pd.DataFrame(np.load(filepath.format('train', signal))).reset_index()
            X_train.columns = [str(col) for col in X_train.columns]

            X_test = pd.DataFrame(np.load(filepath.format('test', signal))).reset_index()
            X_test['index'] += X_train['index'].iloc[-1] + 1
            X_test.columns = [str(col) for col in X_test.columns]

            AnomalyDataset(**{
                'source': cls.source,
                'dataset': dataset,
                'signal': signal,

                'train': Dataset(**{
                    'name': 'train',
                    'data': X_train,
                }),
                'test': Dataset(**{
                    'name': 'test',
                    'data': X_test,
                }),
                'anomalies': anomalies
            }).save(out_directory)
