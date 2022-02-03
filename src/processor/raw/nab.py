import json
import os

import pandas as pd
from tqdm import tqdm

from src.configuration.constants import RAW_DATA_DIRECTORY, INTERIM_DATA_DIRECTORY
from src.processor.dataset import AnomalyDataset, Dataset
from src.processor.raw.processor import RawProcessor


class NABProcessor(RawProcessor):
    source: str = 'NAB'

    @classmethod
    def process(cls, input_directory: str = RAW_DATA_DIRECTORY, out_directory: str = INTERIM_DATA_DIRECTORY):
        anomalies_directory = os.path.join(input_directory, cls.source, 'labels', 'combined_windows.json')
        data_directory = os.path.join(input_directory, cls.source, 'data')
        with open(anomalies_directory, 'rb') as f:
            anomalies_json = json.load(f)

        for key, value in tqdm(anomalies_json.items()):
            dataset, filename = key.split('/')
            signal = filename.split('.')[0]
            anomalies = pd.DataFrame(value, columns=['start', 'end'])
            anomalies['start'] = pd.to_datetime(anomalies['start']).values.astype(int)
            anomalies['end'] = pd.to_datetime(anomalies['end']).values.astype(int)

            filepath = os.path.join(data_directory, dataset, filename)
            df = pd.read_csv(filepath)
            df = df.rename(columns={'timestamp': 'index'})
            df['index'] = pd.to_datetime(df['index']).values.astype(int)
            df['labels'] = cls.create_index_labels(df['index'], anomalies)

            AnomalyDataset(**{
                'source': cls.source,
                'dataset': dataset,
                'signal': signal,

                'train': Dataset(**{
                    'name': 'train',
                    'data': df,
                }),
                'test': Dataset(**{
                    'name': 'test',
                    'data': df,
                }),
                'anomalies': anomalies
            }).save(out_directory)
