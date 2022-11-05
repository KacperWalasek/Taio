import argparse
import logging
import os.path
import shutil
from http.client import HTTPResponse
from io import BytesIO, TextIOWrapper
from typing import List, TextIO, Union
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from scipy.io.arff import loadarff


class DataConverter:
    def __init__(self, args: List[str] = None):
        parser = self._get_argparser()
        self.args = parser.parse_args(args)
        logging.basicConfig(level=self.args.loglevel)
        self.convert()

    def convert(self) -> None:
        with self._get_zip() as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                self._process_arff_file(zfile, "Train")
                self._process_arff_file(zfile, "Test")

    def _get_zip(self) -> Union[HTTPResponse, TextIO]:
        zipurl = f'https://www.timeseriesclassification.com/Downloads/{self.args.dataset_name}.zip'
        return urlopen(zipurl) if not self.args.src_dir else open(
            os.path.join(self.args.src_dir, f"{self.args.dataset_name}.zip"), "rb")

    def _process_arff_file(self, zfile, suffix: str) -> None:
        in_mem_fo = TextIOWrapper(zfile.open(self.__get_data_filename(suffix)), encoding='utf-8')
        data, meta = loadarff(in_mem_fo)
        categories = set()
        for i, data_item in enumerate(data):
            if data_item[0].shape:
                series = data_item[0]
                category = data_item[1]
            else:
                series = data_item[meta.names()[:-1]]
                category = data_item[meta.names()[-1]]
            category = category.decode()
            category_dir = self.__get_category_dir_path(suffix, category)
            if category not in categories:
                if os.path.exists(category_dir):
                    shutil.rmtree(category_dir)
                os.makedirs(category_dir)
                categories.add(category)
            series_array = np.array(series.tolist()).view(float).T
            np.savetxt(os.path.join(category_dir, f"{i}.csv"), series_array,
                       delimiter=",")
        logging.info(f"Shape of the series: {series_array.shape}")
        logging.info(f"Processed {i + 1} {suffix} files")

    def __get_data_filename(self, suffix: str) -> str:
        return f'{self.args.dataset_name}_{suffix.upper()}.arff'

    def __get_category_dir_path(self, suffix: str, category: str) -> str:
        return os.path.join(self.args.dst_dir, self.args.dataset_name, suffix, category)

    @staticmethod
    def _get_argparser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="data_converter",
                                         description="Utility to convert .arff data to suitable .csv format with an "
                                                     "appropriate structure")
        parser.add_argument('--src-dir',
                            help="a directory in which a file name.zip is placed, if not provided, the data will be "
                                 "downloaded")

        parser.add_argument("--dst-dir", type=str, default="data",
                            help="a destination directory in which preprocessed data will be placed")

        parser.add_argument(
            '-v', '--verbose',
            help="be verbose",
            action="store_const", dest="loglevel", const=logging.INFO,
        )

        parser.add_argument("dataset_name", type=str)

        return parser
