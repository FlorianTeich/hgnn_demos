"""
Testdaten
"""
import datetime
import pathlib
import random
import string
import typing

import numpy as np
import pandas as pd
import sqlalchemy
from faker import Faker
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sqlalchemy import create_engine

from . import utils

fake = Faker("en_US")


def sample_random_dates(
    start=pd.to_datetime("1900-01-01"),
    end=pd.to_datetime("2023-01-01"),
    count: int = 100,
) -> pd.DataFrame:
    """Generate random dates
    """
    if not count > 0:
        raise AssertionError
    start_u = start.value // 10**9
    end_u = end.value // 10**9
    return pd.to_datetime(
        np.random.randint(start_u, end_u, count, dtype=np.int64), unit="s"
    )


def sample_unique_pids(count: int = 100) -> typing.List[str]:
    """Generate person IDs.
    """
    if not count > 0:
        raise AssertionError
    ids = random.sample(range(1000000000), count)
    numbers = [str(id_ % 1000000000).zfill(9) for id_ in ids]
    ids = [typing.cast(str, numbers[i]) for i in range(len(numbers))]
    return ids


def sample_genders(count: int = 100) -> np.ndarray:
    """Generate gender data
    """
    if not count > 0:
        raise AssertionError
    possible_genders = ["man", "woman"]
    return np.random.choice(possible_genders, count, p=[0.5, 0.5])


def sample_diagnosis_codes(count: int = 100) -> typing.List[str]:
    """Generate diagnosis codes
    """
    if not count > 0:
        raise AssertionError
    letters = string.ascii_uppercase
    rand_letters = random.choices(letters, k=count)
    ids = np.random.choice(range(100), count)
    numbers = [str(id_ % 10).zfill(1) for id_ in ids]
    complete_codes = [
        rand_letters[i] + numbers[i] for i in range(count)
    ]
    return complete_codes


def sample_drug_codes(count: int = 100) -> typing.List[str]:
    """Generate drug codes
    """
    if not count > 0:
        raise AssertionError
    letters = string.ascii_uppercase
    rand_letter1 = random.choices(letters, k=count)
    rand_letter2 = random.choices(letters, k=count)
    complete_dcs = [
        rand_letter1[i]
        + rand_letter2[i]
        for i in range(count)
    ]
    return complete_dcs


def generate_dataframe_drug_consumption(
    unique_pids: typing.List[str], count: int = 1000
) -> pd.DataFrame:
    """Generate drug consumption table
    """
    if not count > 0:
        raise AssertionError
    pids = np.random.choice(unique_pids, size=count, replace=True).tolist()
    dc = sample_drug_codes(count=count)
    dates = sample_random_dates(count=count)
    return pd.DataFrame(
        {"PID": pids, "DrugCode": dc, "DrugDate": dates,}
    )


def generate_dataframe_persons(count: int = 100) -> pd.DataFrame:
    """Generate person table
    """
    if not count > 0:
        raise AssertionError

    ids = sample_unique_pids(count)
    gender = sample_genders(count)
    birth_dates = sample_random_dates(
        count=count,
    )

    return pd.DataFrame(
        {
            "PID": ids,
            "Gender": gender,
            "Birthday": birth_dates,
        }
    )


def generate_dataframe_diagnoses(
    unique_pids: typing.List[str], count: int = 100
) -> pd.DataFrame:
    """Generate Diagnoses
    """
    if not count > 0:
        raise AssertionError
    pids = np.random.choice(unique_pids, size=count, replace=True).tolist()
    icds = sample_diagnosis_codes(count=count)
    date = sample_random_dates(count=count)
    
    return pd.DataFrame(
        {
            "PID": pids,
            "DiagnosisCode": icds,
            "DiagnosisDate": date,
        }
    )



def generate_all_tables(
    person_count: int = 100, conn=sqlalchemy.engine.base.Engine
) -> None:
    """Generate all tables
    """
    people = generate_dataframe_persons(count=person_count)
    drugs = generate_dataframe_drug_consumption(
        unique_pids=people["PID"].tolist(), count=10 * person_count
    )
    diagnoses = generate_dataframe_diagnoses(
        unique_pids=people["PID"].tolist(), count=10 * person_count
    )

    connection_session = conn.connect()
    people.to_sql("persons", connection_session, if_exists="replace")
    drugs.to_sql(
        "drugs", connection_session, if_exists="replace"
    )
    diagnoses.to_sql("diagnoses", connection_session, if_exists="replace")

def generate_toy_datasamples(num_samples=1000):
    sqlite_file_path = str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/backend.db"
    print(sqlite_file_path)
    backend = create_engine("sqlite:///" + sqlite_file_path)
    generate_all_tables(person_count=num_samples, conn=backend)
    persons = pd.read_sql_table("persons", con=backend)
    diagnosis_data = pd.read_sql_table("diagnoses", con=backend)
    drug_data = pd.read_sql_table("drugs", con=backend)

    train, validate, test = np.split(persons.sample(frac=1), [int(.6*len(persons)), int(.8*len(persons))])
    persons.loc[train.index, "mode"] = "train"
    persons.loc[validate.index, "mode"] = "validate"
    persons.loc[test.index, "mode"] = "test"

    # Get gender and age
    scaler2 = StandardScaler()
    le = preprocessing.LabelEncoder()
    le.fit(persons[persons["mode"] != "test"].Gender)
    persons["gender_encoded"] = le.fit_transform(persons.Gender)

    persons["now"] = pd.to_datetime(datetime.datetime.now())
    persons["age"] = (persons.now - persons.Birthday).dt.days
    persons["age"] = scaler2.fit_transform(np.expand_dims(pd.to_numeric(persons["age"], downcast="float"), 1))

    # Get diagnosis codes
    scaler = MaxAbsScaler()
    diagnosis_data = diagnosis_data.merge(persons[["PID", "Birthday", "mode"]], on="PID", how="left")
    diagnosis_data["diagnosis_age"] = (diagnosis_data["DiagnosisDate"] - diagnosis_data.Birthday).dt.days
    scaler.fit(diagnosis_data[diagnosis_data["mode"] != "test"][["diagnosis_age"]])
    diagnosis_data["diagnosis_age"] = scaler.transform(diagnosis_data[["diagnosis_age"]])
    diagnosis_data["diagnosis_short"] = diagnosis_data["DiagnosisCode"].str[:1].apply(str.lower)
    diagnosis_agg = diagnosis_data.groupby("PID")['diagnosis_short'].apply(" ".join)

    merged_diagnosis = persons.merge(diagnosis_agg, how="left", on="PID")

    # Get drug codes
    scaler2 = MaxAbsScaler()
    drug_data = drug_data.merge(persons[["PID", "Birthday", "mode"]], on="PID", how="left")
    drug_data["drug_age"] = (drug_data["DrugDate"] - drug_data.Birthday).dt.days
    scaler2.fit(drug_data[drug_data["mode"] != "test"][["drug_age"]])
    drug_data["drug_age"] = scaler2.transform(drug_data[["drug_age"]])
    drug_data["drug_short"] = drug_data["DrugCode"].str[:1].apply(str.lower)
    drug_agg = drug_data.groupby("PID")['drug_short'].apply(" ".join)

    merged_drug = persons.merge(drug_agg, how="left", on="PID")

    def analyzer_custom(doc):
        return doc.split()

    vectorizer_diagnosis = CountVectorizer(vocabulary=list(string.ascii_lowercase),
                                analyzer=analyzer_custom)
    vectorizer_drug = CountVectorizer(vocabulary=list(string.ascii_lowercase),
                                analyzer=analyzer_custom)

    data_diagnosis = vectorizer_diagnosis.transform(merged_diagnosis["diagnosis_short"].values.astype('U')).todense()
    data_drug = vectorizer_drug.transform(merged_drug["drug_short"].values.astype('U')).todense()
    merged_diagnosis.loc[:, "diagnosis_vec"] = data_diagnosis.tolist()
    merged_drug.loc[:, "drug_vec"] = data_drug.tolist()

    X = np.hstack([merged_diagnosis[["gender_encoded", "age"]].values, 
                np.vstack(merged_diagnosis["diagnosis_vec"].values),
                np.vstack(merged_drug["drug_vec"].values)])

    # Create Target Labels
    diagnosis_data["label_i"] = ((diagnosis_data["diagnosis_short"] <= "k") & (diagnosis_data["diagnosis_age"] < 0))
    drug_data["label_a"] = ((drug_data["drug_short"] <= "k") & (drug_data["drug_age"] > 0))
    persons = persons.merge(diagnosis_data[["PID", "label_i"]].groupby("PID").max(), on="PID", how="left")
    persons = persons.merge(drug_data[["PID", "label_a"]].groupby("PID").max(), on="PID", how="left")
    persons["label"] = (persons["label_i"] == True) & (persons["label_a"] == True)
    print(persons.label.value_counts())

    data = np.hstack([utils.encode_strings(diagnosis_data["diagnosis_short"]), diagnosis_data[["diagnosis_age"]].values])
    feats = pd.DataFrame(((x,) for x in data), columns=["features"])
    diagnosis_data = pd.concat([diagnosis_data, feats], axis=1)

    data = np.hstack([utils.encode_strings(drug_data["drug_short"]), drug_data[["drug_age"]].values])
    feats = pd.DataFrame(((x,) for x in data), columns=["features"])
    drug_data = pd.concat([drug_data, feats], axis=1)

    data = persons[["age", "gender_encoded"]].values
    feats = pd.DataFrame(((x,) for x in data), columns=["features"])
    persons = pd.concat([persons, feats], axis=1)

    persons.to_parquet(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/persons.parquet")
    diagnosis_data.to_parquet(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/diagnoses.parquet")
    drug_data.to_parquet(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/drugs.parquet")

    train_inds = pd.DataFrame({"ids": persons[persons["mode"] == "train"].index.tolist()})
    test_inds = pd.DataFrame({"ids": persons[persons["mode"] == "test"].index.tolist()})
    train_inds.to_parquet(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/train_ids.parquet")
    test_inds.to_parquet(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/test_ids.parquet")
