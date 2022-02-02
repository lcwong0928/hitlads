import numpy as np
from mlprimitives import load_primitive

from src.configuration.constants import PROCESSED_DATA_DIRECTORY, INTERIM_DATA_DIRECTORY
from src.processor.dataset import AnomalyDataset


class MSLProcessor:
    source: str = 'NASA'
    dataset: str = 'MSL'

    def __init__(self, signal: str) -> None:
        self.signal = signal
        self.anomaly_dataset = AnomalyDataset.load(self.source, self.dataset, self.signal, INTERIM_DATA_DIRECTORY)
        self.primitives = []

    def fit(self):
        X = self.anomaly_dataset.train.data

        # 1. Create an equi-spaced time series by aggregating values over fixed specified interval
        params = {
            "time_column": "index",
            "interval": 1,
            "method": "mean"
        }
        primitive = load_primitive('mlprimitives.custom.timeseries_preprocessing.time_segments_aggregate',
                                   arguments=params)
        self.primitives.append(primitive)
        X, index = primitive.produce(X=X)

        # 2. Imputation transformer for filling missing values.
        params = {
            'X': X
        }
        primitive = load_primitive('sklearn.impute.SimpleImputer', arguments=params)
        primitive.fit()
        self.primitives.append(primitive)
        X = primitive.produce(X=X)

        # 3. Transforms features by scaling each feature to a given range.
        params = {
            "feature_range": [-1, 1],
            'X': X,
        }
        primitive = load_primitive('sklearn.preprocessing.MinMaxScaler', arguments=params)
        primitive.fit()
        self.primitives.append(primitive)

        # 4. Uses a rolling window approach to create the sub-sequences out of time series data
        params = {
            "target_column": 0,
            "window_size": 100,
            'target_size': 1,
            'step_size': 1
        }
        primitive = load_primitive('mlprimitives.custom.timeseries_preprocessing.rolling_window_sequences',
                                   arguments=params)
        self.primitives.append(primitive)
        X, _, index, _ = primitive.produce(X=X, index=index)

        # 5. Get target (y)
        params = {
            "target_index": 0,
            "axis": 2
        }
        primitive = load_primitive('orion.primitives.timeseries_preprocessing.slice_array_by_dims',
                                   arguments=params)
        self.primitives.append(primitive)
        y = primitive.produce(X=X)

        labels = (np.sum(X[:, :, -1], axis=1) > 50).astype(int)
        X = X[:, :, :-1]

        self.anomaly_dataset.train.update(**{
            'index': index,
            'X': X,
            'y': y,
            'labels': labels,
        })

        return self

    def transform(self):
        X = self.anomaly_dataset.test.data
        X, index = self.primitives[0].produce(X=X)
        X = self.primitives[1].produce(X=X)
        X = self.primitives[2].produce(X=X)
        X, _, index, _ = self.primitives[3].produce(X=X, index=index)
        y = self.primitives[4].produce(X=X)

        labels = (np.sum(X[:, :, -1], axis=1) > 50).astype(int)
        X = X[:, :, :-1]

        self.anomaly_dataset.test.update(**{
            'index': index,
            'X': X,
            'y': y,
            'labels': labels,
        })
        return self

    def fit_transform(self):
        self.fit()
        self.transform()
        self.anomaly_dataset.save(PROCESSED_DATA_DIRECTORY)
        return self
