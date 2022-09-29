import argparse
import configparser
import logging
from typing import List, Literal

from classifier_pipeline.data_reader import DataReader
from classifier_pipeline.ensemble_classifier import EnsembleClassifier
from classifier_pipeline.get_root import get_root
from classifier_pipeline.results_saver import ResultsSaver


class ClassifierPipeline:
    def __init__(self, args: List[str] = None):
        parser = self._get_argparser()
        self.args = parser.parse_args(args)
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=self.args.loglevel,
            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger("main")
        self.logger.info("ClassifierPipeline init")

    def run(self):
        self.logger.info("Reading data...")
        data_reader = self.get_data_reader(self.args.data_dir, self.args.dataset)
        data_train = data_reader.read_preprocess_train()
        data_test = data_reader.read_preprocess_test()
        self.logger.info("Done reading data")

        classifier_config = self.get_classifier_config(f"{self.args.config_dir}/{self.args.config}")
        self.logger.info("Building and fitting ensemble classifier...")
        classifier = self.build_classifier(self.args.method, classifier_config, self.logger)
        classifier.fit(data_train)
        self.logger.info("Done building and fitting ensemble classifier")

        self.logger.info("Evaluating classifier...")
        train_acc = classifier.evaluate(data_train)
        test_acc = classifier.evaluate(data_test)
        self.logger.info(f"Done evaluating classifier, {train_acc=}, {test_acc=}")

        self.logger.info("Saving the results")
        results_saver = self.get_results_saver(self.args.results_dir, self.args.dataset)
        results_saver.save(self.args.method, classifier_config, train_acc, test_acc)
        self.logger.info("Done saving the results")

    @staticmethod
    def get_data_reader(dir_name: str, dataset_name: str) -> DataReader:
        return DataReader(dir_name, dataset_name)

    @staticmethod
    def get_classifier_config(config_path: str) -> configparser.ConfigParser:
        ret = configparser.ConfigParser()
        ret.read(get_root() / config_path)
        return ret

    @staticmethod
    def build_classifier(
            method: Literal['1_vs_all', 'asymmetric_1_vs_1', 'symmetric_1_vs_1', 'combined_symmetric_1_vs_1'],
            classifier_config: configparser.ConfigParser, logger: logging.Logger) -> EnsembleClassifier:
        return EnsembleClassifier.build_classifier(method, classifier_config, logger)

    @staticmethod
    def get_results_saver(dir_name: str, dataset_name: str) -> ResultsSaver:
        return ResultsSaver(dir_name, dataset_name)

    @staticmethod
    def _get_argparser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="classifier_pipeline",
                                         description="Series classification pipeline")
        parser.add_argument('--data-dir', default="data",
                            help="a root directory with series data")
        parser.add_argument('--config-dir', default="configs",
                            help="a root directory with run configs")
        parser.add_argument('--config', required=True, help="name of the config file")
        parser.add_argument('--method', choices=['1_vs_all', 'asymmetric_1_vs_1', 'symmetric_1_vs_1',
                                                 'combined_symmetric_1_vs_1'], required=True)
        parser.add_argument('--test-length-percentages', type=float, nargs="+", default=[1], help="TODO")
        parser.add_argument('--results-dir', default="results",
                            help="a directory in which to put results")
        parser.add_argument("dataset", help="name of the dataset to test")

        parser.add_argument(
            '-v', '--verbose',
            help="be verbose",
            action="store_const", dest="loglevel", const=logging.INFO,
        )

        return parser
