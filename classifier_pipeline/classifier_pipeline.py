class ClassifierPipeline:
    def __init__(self, args: List[str] = None):
        parser = self._get_argparser()
        self.args = parser.parse_args(args)
        logging.basicConfig(level=self.args.loglevel)
        self.run()

    def run(self):
        data_reader = self.get_data_reader()
        data_train = data_reader.read_train(self.args.data_dir, self.args.dataset)
        data_test = data_reader.read_test(self.args.data_dir, self.args.dataset)

        classifier = self.build_classifier(self.args.method)
        classifier.fit(data_train)

        classifier.evaluate(data_train)
        classifier.evaluate(data_test)

    def get_data_reader(self):


    @staticmethod
    def _get_argparser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="classifier_pipeline",
                                         description="Series classification pipeline")
        parser.add_argument('--data-dir', default="data",
                            help="a root directory with series data")
        parser.add_argument('--config-dir', default="configs",
                            help="a root directory with run configs")
        parser.add_argument('--config', help="name of the config file")
        parser.add_argument('--method', choices=['1_vs_all', 'asymmetric_1_vs_1', 'symmetric_1_vs_1',
                                                 'combined_symmetric_1_vs_1'])
        parser.add_argument('--test-length-percentages', type=float, nargs="+")
        parser.add_argument("dataset", help="name of the dataset to test")

        parser.add_argument(
            '-v', '--verbose',
            help="be verbose",
            action="store_const", dest="loglevel", const=logging.INFO,
        )

        return parser
