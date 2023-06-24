import wfdb
from data_set_utils.models import ECGRecord


class DataMaker:
    def __init__(self, dir: str) -> None:
        self.__dir = dir

    def create_data_for_record(self, record: ECGRecord, symbol: str):
        pass
