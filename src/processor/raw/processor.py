from abc import abstractmethod

import pandas as pd

from src.configuration.constants import RAW_DATA_DIRECTORY, INTERIM_DATA_DIRECTORY


class RawProcessor:

    @classmethod
    @abstractmethod
    def process(cls, input_directory: str = RAW_DATA_DIRECTORY, out_directory: str = INTERIM_DATA_DIRECTORY):
        pass

    @classmethod
    def create_index_labels(cls, index: pd.DataFrame, anomalies: pd.DataFrame) -> list:
        labels = []

        for i in index:
            label = 0
            for start, end in zip(anomalies.start, anomalies.end):
                if start <= i <= end:
                    label = 1
                    break
            labels.append(label)

        return labels
