from data_set_utils.models import ECGRecord, ECGSegment, ECGAnnotation
import numpy as np
from utils.common_utils import (
    remove_baseline_wander,
    remove_powerline_interference,
    high_pass_filter,
    hanning_filter,
)
from ecgdetectors import Detectors
import time


class DataMaker:
    def __init__(
        self,
        records: list[ECGRecord],
        powerline_freq: float,
        sampling_freq: float,
        signal_name: str,
        hanning_window_filter_size: int,
        high_pass_filter_freq: float,
        error: float,
        before_peak: int,
        after_peak: int,
    ) -> None:
        self.__records = records
        self.__powerline_freq = powerline_freq
        self.__sampling_freq = sampling_freq
        self.__signal_name = signal_name
        self.__hanning_window_filter_size = hanning_window_filter_size
        self.__high_pass_filter_freq = high_pass_filter_freq
        self.__error = error
        self.__before_peak = before_peak
        self.__after_peak = after_peak

    @staticmethod
    def get_lead_signal(record: ECGRecord, signal_name: str) -> np.ndarray:
        signal_names = np.array(record.signals())
        try:
            signal_index = np.where(signal_names == signal_name)[0][0]
        except IndexError:
            return np.array([])
        ecg_signal = record.ecg_signal()
        lead_signal = ecg_signal[:, signal_index]
        return lead_signal

    @staticmethod
    def clean_signal(
        signal: np.ndarray,
        powerline_frequency: float = 50,
        high_pass_filter_freq: float = 0.5,
        hanning_filter_window_size: int = 15,
        sampling_freq: float = 360,
        error: float = 2,
    ):
        new_signal_after_baseline_removal = remove_baseline_wander(np.copy(signal))
        new_signal_after_removal_p_intr = remove_powerline_interference(
            powerline_frequency,
            new_signal_after_baseline_removal,
            sampling_rate=sampling_freq,
            error=error,
        )
        filtered_high_pass = high_pass_filter(
            new_signal_after_removal_p_intr, high_pass_filter_freq
        )
        filtered_low_pass = hanning_filter(
            filtered_high_pass, hanning_filter_window_size
        )
        return np.copy(filtered_low_pass)

    @staticmethod
    def segment_signal(
        before_peak: int,
        after_peak,
        signal: np.ndarray,
        record_id: str,
        sampling_freq: float,
    ) -> list[ECGSegment]:
        ecg_segments = list()
        detector = Detectors(sampling_frequency=sampling_freq)
        r_peaks = detector.pan_tompkins_detector(signal)
        for i in range(0, len(r_peaks)):
            segment_start = max(r_peaks[i] - before_peak, 0)
            segment_end = min(r_peaks[i] + after_peak, len(signal))
            ecg_segment = ECGSegment(
                record_id,
                segment_start,
                segment_end,
                readings=signal[segment_start:segment_end],
                target_classes=list(),
            )
            ecg_segments.append(ecg_segment)

        return ecg_segments

    @staticmethod
    def add_target_classes(
        annotation_symbols: np.ndarray,
        annotation_indexes,
        segments: list[ECGSegment],
        record_id: str,
    ) -> None:
        for annotation_symbol, annotation_index in zip(
            annotation_symbols, annotation_indexes
        ):
            annotation = ECGAnnotation(annotation_index, annotation_symbol)
            for segment in segments:
                start, end = segment.get_segment_positions()
                if (
                    annotation.get_pos() >= start
                    and annotation.get_pos() <= end
                    and segment.get_record_id() == record_id
                ):
                    segment.add_annotation(annotation)
                    break

    def perform_segmentation(self, symbols: list[str]) -> list[ECGSegment]:
        segments = []
        for record in self.__records:
            signal = DataMaker.get_lead_signal(record, self.__signal_name)
            if not len(signal) > 0:
                print(f"No {self.__signal_name} for {record.record_name()}")
                continue
            clean_signal = DataMaker.clean_signal(
                signal,
                self.__powerline_freq,
                self.__high_pass_filter_freq,
                self.__hanning_window_filter_size,
                self.__sampling_freq,
                self.__error,
            )
            segments_for_record = DataMaker.segment_signal(
                self.__before_peak,
                self.__after_peak,
                clean_signal,
                record.record_name(),
                self.__sampling_freq,
            )
            DataMaker.add_target_classes(
                record.annotations_present(),
                record.annotation_indexes(),
                segments_for_record,
                record.record_name(),
            )
            segments.extend(segments_for_record)
        return segments
