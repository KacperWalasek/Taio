import configparser

import pandas as pd

from classifier_pipeline.get_root import get_root


class ResultsSaver:
    """
    Class for saving the results
    """

    def __init__(self, dir_name: str, dataset_name: str):
        self.filepath = get_root() / dir_name / f"{dataset_name}.csv"

    def save(self, method: str, config: configparser.ConfigParser, train_acc: float, test_acc: float):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        config_params_dict = dict((f"{s.lower()}_{k}", v) for s in config.sections() for k, v in config.items(s))
        df = pd.DataFrame([{
            "method": method,
            "train_acc": train_acc,
            "test_acc": test_acc,
            **config_params_dict
        }]).apply(pd.to_numeric, errors='ignore')
        df.to_csv(self.filepath, mode='a', index=False)
