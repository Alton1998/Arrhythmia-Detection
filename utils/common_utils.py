import os

def load_patient_ids(dir:str)->list:
    filtered_files = filter(lambda x: x.endswith(".atr"),os.listdir(dir))
    patient_ids = map(lambda x: x.replace(".atr",""),filtered_files)
    return list(patient_ids)