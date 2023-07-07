"""
Testdaten
"""
import sqlalchemy
import random
import string
import typing

import numpy as np
import pandas as pd
from faker import Faker

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
