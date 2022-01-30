import numpy as np
from mlprimitives import load_primitive

from src.dataset.interim_dataset import InterimDataset


class MSLProcessor:
    source = 'NASA'
    dataset = 'MSL'

    def __init__(self, signal: str):
        self.signal = signal
        self.dataset = InterimDataset.load(self.source, self.dataset, self.signal)
        self.primitives = []

    def fit(self, X, index):
        # 1. Imputation transformer for filling missing values.
        params = {
            'X': X
        }
        primitive = load_primitive('sklearn.impute.SimpleImputer', arguments=params)
        primitive.fit()
        self.primitives.append(primitive)
        X = primitive.produce(X=X)

        # 2. Transforms features by scaling each feature to a given range.
        params = {
            "feature_range": [-1, 1],
            'X': X,
        }
        primitive = load_primitive('sklearn.preprocessing.MinMaxScaler', arguments=params)
        primitive.fit()
        self.primitives.append(primitive)

        # 3. Uses a rolling window approach to create the sub-sequences out of time series data
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

        y = np.expand_dims(X[:, :, 0], 2)

        return X, y, index

    def transform(self, X, index):
        X = self.primitives[0].produce(X=X)
        X = self.primitives[1].produce(X=X)
        X, _, index, _ = self.primitives[2].produce(X=X, index=index)
        y = np.expand_dims(X[:, :, 0], 2)
        return X, y, index
