import wfdb
import copy
from data_set_utils.data_set_exceptions import DataSetReaderException


class DataReader:
    def __init__(self, dir: str):
        self.__dir = dir

    def get_dir(self):
        return copy.copy(self.dir)

    def __record_path(self, record_id: str) -> str:
        return self.__dir + "/" + record_id

    def plot_record(self, record_id: str, sampto: int):
        path = self.__record_path(record_id)
        try:
            record = wfdb.rdrecord(path, sampto=sampto)
            annotation = wfdb.rdann(path, "atr", sampto=sampto)
            wfdb.plot_wfdb(record=record, annotation=annotation, plot_sym=True)
            return record
        except Exception as e:
            raise DataSetReaderException(f"Record {record_id} is not good", e)

    def plot_records(self, record_ids: list, sampto: int):
        for record_id in record_ids:
            try:
                self.plot_record(record_id, sampto)
            except DataSetReaderException as e:
                print(e)

    def read_header(self, record_id: str):
        path = self.__record_path(record_id)
        return wfdb.rdheader(path)

    def plot_annotation_signal(self, record_id: str, sampto: int):
        path = self.__record_path(record_id)
        try:
            record = wfdb.rdrecord(path, sampto=sampto)
            annotation = wfdb.rdann(path, "atr", sampto=sampto)
            wfdb.plot_items(
                signal=record.p_signal,
                ann_samp=[annotation.sample, annotation.sample],
                ann_sym=[annotation.symbol, annotation.symbol],
            )
        except Exception as e:
            raise DataSetReaderException(f"Record {record_id} is not good", e)

    def plot_annotation_signals(self, record_ids: list, sampto: int):
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
