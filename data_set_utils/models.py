from wfdb import Record, MultiRecord, Annotation
from typing import Union
import numpy as np
from utils.common_utils import calculate_index
from typing import Iterable,Tuple


class ECGRecord:
    def __init__(
        self,
        record: Union[Record, MultiRecord],
        annotation: Annotation,
        bit_depth: int,
        max_voltage: float,
        min_voltage: float,
    ) -> None:
        self.__record: Union[Record, MultiRecord] = record
        self.__annotation: Annotation = annotation
        self.__bit_depth: int = bit_depth
        self.__max_voltage: float = max_voltage
        self.__min_voltage: float = min_voltage
        self.__levels: int = 2**self.__bit_depth
        self.__step_size: float = (
            self.__max_voltage - self.__min_voltage
        ) / self.__levels

    def record_name(self) -> str:
        return self.__record.record_name

    def signals(self) -> list[str]:
        return self.__record.sig_name

    def signal_number(self) -> int:
        return self.__record.n_sig

    def sampling_frequency(self) -> int:
        return self.__record.fs

    def signal_length(self) -> int:
        return self.__record.sig_len

    def patient_age(self) -> int:
        age: int = int(self.__record.comments[0].split(" ")[0])
        return age

    def patient_gender(self) -> str:
        gender: str = self.__record.comments[0].split(" ")[1]
        return gender

    def comments(self) -> list[str]:
        return self.__record.comments

    def ecg_signal(self) -> np.ndarray:
        return self.__record.p_signal

    def ecg_signal_discretized(self) -> np.ndarray:
        calculate_discrete = np.vectorize(
            lambda x: calculate_index(x, self.__min_voltage, self.__step_size)
        )
        return calculate_discrete(np.copy(self.__record.p_signal))

    def annotations_present(self) -> list[str]:
        return self.__annotation.symbol

    def annotation_indexes(self) -> list[int]:
        return self.__annotation.sample

    def leads(self) -> list[str]:
        return self.__record.sig_name


class ECGAnnotation:
    def __init__(self,pos:int,symbol:str) -> None:
        self.__pos = pos
        self.__symbol = symbol
    
    def get_pos(self)->int:
        return self.__pos
    
    def get_symbol(self)->str:
        return self.__symbol
    def __str__(self) -> str:
        return str("Pos:" + str(self.__pos) + "Symbol:" + self.__symbol)
    
class ECGSegment:
    def __init__(
        self, record_id: str, segment_start: int, segment_end: int, readings:np.ndarray ,target_classes: Iterable[ECGAnnotation]
    ) -> None:
        self.__record_id = record_id
        self.__segment_start = segment_start
        self.__segment_end = segment_end
        self.__target_classes = target_classes
        self.__readings = readings
    
    def get_record_id(self) ->str:
        return self.__record_id
    
    def get_segment_positions(self)->Tuple[int,int]:
        return (self.__segment_start,self.__segment_end)
    

    def get_readings(self)->str:
        return self.__readings
    
    def __str__(self) -> str:
        return "Record Id: " + self.__record_id + "\n" + "Start:" + str(self.__segment_start) + "\n" + "End:" + str(self.__segment_end) + "\n"
    
    def get_annotations(self)->list[Annotation]:
        return self.__target_classes
    
    def add_annotation(self,annotion:ECGAnnotation) -> None:
        self.__target_classes.append(annotion)