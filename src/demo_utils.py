import numpy as np
import pandas as pd


def edges_diagnosis():
    persons = pd.read_parquet("./data/persons.parquet")
    diagnosis_data = pd.read_parquet("./data/diagnoses.parquet")
    return np.flip(persons.merge(diagnosis_data, on="PID", how="inner")[["index_x", "index_y"]].values.T)

def edges_drugs():
    persons = pd.read_parquet("./data/persons.parquet")
    drugs_data = pd.read_parquet("./data/diagnoses.parquet")
    return np.flip(persons.merge(drugs_data, on="PID", how="inner")[["index_x", "index_y"]].values.T)
