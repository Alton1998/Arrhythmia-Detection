import wfdb
import copy
from data_set_utils.data_set_exceptions import DataSetReaderException
from wfdb import Record, MultiRecord, Annotation
from typing import Union
from data_set_utils.models import ECGRecord
import numpy as np


def count_annotations(record_list: list[ECGRecord]) -> tuple[np.ndarray, np.ndarray]:
    annotation_list: list[str] = []
    for record in record_list:
        annotation_list.extend(record.annotations_present())
    annotation_array = np.array(annotation_list)
    return np.unique(annotation_array, return_counts=True)


class DataReader:
    def __init__(
        self, dir: str, bit_depth: int, max_voltage: float, min_voltage: float
    ) -> None:
        self.__dir: str = dir
        self.__bit_depth: int = bit_depth
        self.__max_voltage: float = max_voltage
        self.__min_voltage: float = min_voltage

    def get_dir(self) -> str:
        return copy.copy(self.dir)

    def __record_path(self, record_id: str) -> str:
        return self.__dir + "/" + record_id

    def plot_record(self, record_id: str, sampto: int, sampfrom: int = 0) -> None:
        path: str = self.__record_path(record_id)
        try:
            record: Union[Record, MultiRecord] = wfdb.rdrecord(
                path, sampfrom=sampfrom, sampto=sampto
            )
            annotation: Annotation = wfdb.rdann(
                path, "atr", sampfrom=sampfrom, sampto=sampto,shift_samps=True
            )
            wfdb.plot_wfdb(record=record, annotation=annotation, plot_sym=True)
        except Exception as e:
            raise DataSetReaderException(f"Record {record_id} is not good", e)

    def plot_records(
        self, record_ids: list[str], sampto: int, sampfrom: int = 0
    ) -> None:
        for record_id in record_ids:
            try:
                self.plot_record(record_id, sampto, sampfrom)
            except DataSetReaderException as e:
                print(e)

    def read_header(self, record_id: str) -> Union[Record, MultiRecord]:
        path: str = self.__record_path(record_id)
        return wfdb.rdheader(path)

    def plot_annotation_signal(self, record_id: str, sampto: int):
        path: str = self.__record_path(record_id)
        try:
            record: Union[Record, MultiRecord] = wfdb.rdrecord(path, sampto=sampto)
            annotation: Annotation = wfdb.rdann(path, "atr", sampto=sampto)
            wfdb.plot_items(
                signal=record.p_signal,
                ann_samp=[annotation.sample, annotation.sample],
                ann_sym=[annotation.symbol, annotation.symbol],
            )
        except Exception as e:
            raise DataSetReaderException(f"Record {record_id} is not good", e)

    def load_record(self, record_id: str) -> ECGRecord:
        path: str = self.__record_path(record_id)
        try:
            record: Union[Record, MultiRecord] = wfdb.rdrecord(path)
            annotation: Annotation = wfdb.rdann(path, "atr")
            return ECGRecord(
                record,
                annotation,
                self.__bit_depth,
                self.__max_voltage,
                self.__min_voltage,
            )
        except Exception as e:
            raise DataSetReaderException(f"Record {record_id} is not good", e)

    def load_records(self, record_ids: list[str]) -> list[ECGRecord]:
        listrecord_list: list[ECGRecord] = []
        for record_id in record_ids:
            try:
                listrecord_list.append(self.load_record(record_id))
            except DataSetReaderException as e:
                print(e)

        return listrecord_list

    def plot_annotation_signals(self, record_ids: list[str], sampto: int):
        for record_id in record_ids:
            try:
                self.plot_annotation_signal(record_id, sampto)
            except DataSetReaderException as e:
                print(e)


def main():
    dr = DataReader(
        "/mnt/d/git/Arrhythmia-Detection/raw_data/mit-bih-arrhythmia-database-1.0.0"
    )
    dr.plot_record("100", 1500)


if __name__ == "__main__":
    main()
