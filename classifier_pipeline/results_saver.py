import configparser
from collections.abc import MutableMapping
from pathlib import Path

import pandas as pd


def get_root() -> Path:
    return Path(__file__).parent.parent.resolve()


class ResultsSaver:
    """
    Class for saving the results
    """

    def __init__(self, dir_name: str, dataset_name: str):
        self.filepath = get_root() / dir_name / f"{dataset_name}.csv"

    def save(self, method: str, config: MutableMapping, train_acc: float, test_acc: float):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        config_params_dict = dict((f"{s.lower()}_{k}", v) for s in config.sections() for k, v in config.items(s))
        df = pd.DataFrame([{
            "method": method,
            "train_acc": train_acc,
            "test_acc": test_acc,
            **config_params_dict
        }])
        df.to_csv(self.filepath, mode='a', index=False)
