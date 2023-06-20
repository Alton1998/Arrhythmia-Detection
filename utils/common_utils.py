import os
import numpy as np


def load_patient_ids(dir: str) -> list:
    filtered_files = filter(lambda x: x.endswith(".atr"), os.listdir(dir))
    patient_ids = map(lambda x: x.replace(".atr", ""), filtered_files)
    return list(patient_ids)


def calculate_index(x: float, xmin: float, delta: float) -> round:
    range: float = np.round(x - xmin, decimals=2)
    index: float = np.round(range / delta, decimals=2)
    return np.round(index)
