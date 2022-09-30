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

    def run(self) -> None:
        self.logger.info("Reading data...")
        data_reader = self.get_data_reader(self.args.data_dir, self.args.dataset)
        data_train = data_reader.read_preprocess_train()
        data_test = data_reader.read_preprocess_test()
        self.logger.info("Done reading data")

        config_filenames: List[str] = self.args.configs if self.args.configs else [self.args.config]

        for rep in range(self.args.num_reps):
            for config_name in config_filenames:
                self.logger.info(f"Starting pipeline, {rep=}, {config_name=}")

                self.logger.info("Reading config...")
                classifier_config = self.get_classifier_config(f"{self.args.config_dir}/{self.args.config}")
                self.logger.info("Building and fitting ensemble classifier...")
                classifier = self.build_classifier(self.args.method, classifier_config, self.logger)
                classifier.fit(data_train)
                self.logger.info("Done building and fitting ensemble classifier")

                for series_length_fraction in self.args.test_length_fractions:
                    self.logger.info(f"Preparing evaluation datasets {series_length_fraction=}...")
                    data_train_truncated = data_train.truncate(series_length_fraction,
                                                               self.get_min_series_length(classifier_config),
                                                               self.logger)
                    data_test_truncated = data_test.truncate(series_length_fraction,
                                                             self.get_min_series_length(classifier_config), self.logger)
                    self.logger.info(f"Done preparing evaluation datasets")

                    self.logger.info("Evaluating classifier...")
                    train_acc = classifier.evaluate(data_train_truncated)
                    test_acc = classifier.evaluate(data_test_truncated)
                    self.logger.info(
                        f"Done evaluating classifier, {series_length_fraction=}, {train_acc=}, {test_acc=}")

                    self.logger.info("Saving the results")
                    results_saver = self.get_results_saver(self.args.results_dir, self.args.dataset)
                    results_saver.save(self.args.method, classifier_config, series_length_fraction, train_acc, test_acc)
                    self.logger.info(f"Done saving the results, {series_length_fraction=}")

                self.logger.info(f"Finished pipeline, {rep=}, {config_name=}")

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
    def get_min_series_length(config: configparser.ConfigParser) -> int:
        return config.getint("BaseClassifier", "MovingWindowSize")

    @staticmethod
    def _get_argparser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="classifier_pipeline",
                                         description="Series classification pipeline")
        parser.add_argument('--data-dir', default="data",
                            help="a root directory with series data")
        parser.add_argument('--config-dir', default="configs",
                            help="a root directory with run configs")
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--config', help="name of the config file")
        group.add_argument('--configs', nargs='+', default=None, help="name of the configs file")
        parser.add_argument('--method', default=None, choices=['1_vs_all', 'asymmetric_1_vs_1', 'symmetric_1_vs_1',
                                                               'combined_symmetric_1_vs_1'], required=True)
        parser.add_argument('--test-length-fractions', type=float, nargs="+", default=[1],
                            help="series length fractions for early classification to verify")
        parser.add_argument('--results-dir', default="results",
                            help="a directory in which to put results")
        parser.add_argument(
            '-v', '--verbose',
            help="be verbose",
            action="store_const", dest="loglevel", const=logging.INFO,
        )
        parser.add_argument('--num-reps', type=int, default=5, help="a number of repetitions of each experiment")

        parser.add_argument("dataset", help="name of the dataset to test")

        return parser
