import configparser

import pandas as pd

from classifier_pipeline.get_root import get_root


class ResultsSaver:
    """
    Class for saving the results
    """

    def __init__(self, dir_name: str, dataset_name: str):
        """

        @param dir_name:
        @param dataset_name:
        """
        self.filepath = get_root() / dir_name / f"{dataset_name}.csv"

    def save(self, method: str, config: configparser.ConfigParser, series_length_fraction: float, train_acc: float,
             test_acc: float) -> None:
        """

        @param method:
        @param config:
        @param series_length_fraction:
        @param train_acc:
        @param test_acc:
        @return:
        """
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        config_params_dict = dict((f"{s.lower()}_{k}", v) for s in config.sections() for k, v in config.items(s))
        df = pd.DataFrame([{
            "method": method,
            "series_length_fraction": series_length_fraction,
            "train_acc": train_acc,
            "test_acc": test_acc,
            **config_params_dict
        }]).apply(pd.to_numeric, errors='ignore')
        df.to_csv(self.filepath, mode='a', index=False,
                  header=not (self.filepath.exists() and self.filepath.stat().st_size > 0))
