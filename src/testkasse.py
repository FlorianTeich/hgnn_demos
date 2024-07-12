"""
Testmodul
"""

import datetime
import random
import string
import typing

import faker
import numpy as np
import pandas as pd
import sqlalchemy
from faker import Faker

fake = Faker("de_DE")


def sample_random_dates(
    start=pd.to_datetime("1900-01-01"),
    end=pd.to_datetime("2022-01-01"),
    count: int = 100,
) -> pd.DataFrame:
    """Funktion zum Erzeugen zufälliger Datumsangaben.

    Args:
        start (datetime.datetime, optional): Beginn der Zeitspanne aus der zufällige
            Punkte gesampelt werden sollen. Default ist 1900.
        end (datetime.datetime, optional): Ende der Zeitspanne aus der zufällige
            Punkte gesampelt werden sollen. Default ist 2022.
        count (int, optional): Anzahl der zu erzeugenden Zeitpunkte. Default ist 100.
    Returns:
        pd.DataFrame: Ein Dataframe mit allen erzeugten Datumsangaben.
    """
    if not count > 0:
        raise AssertionError
    start_u = start.value // 10**9
    end_u = end.value // 10**9
    return pd.to_datetime(
        np.random.randint(start_u, end_u, count, dtype=np.int64), unit="s"
    )


def sample_unique_kvnrs(count: int = 100) -> typing.List[str]:
    """Funktion zum Erzeugen von zufälligen KVNRs.

    Args:
        count (int, optional): Anzahl der zu erzeugenden KVNRs. Default is 100.

    Returns:
        typing.List[str]: Liste mit allen erzeugten KVNRs.
    """
    if not count > 0:
        raise AssertionError
    ids = random.sample(range(26000000000), count)
    # now convert to strings and substitute the 1000000000-places with characters
    chars = [str(chr(65 + id_ // 1000000000)) for id_ in ids]
    numbers = [str(id_ % 1000000000).zfill(9) for id_ in ids]
    ids = [typing.cast(str, chars[i] + numbers[i]) for i in range(len(numbers))]  # type: ignore
    return ids  # type: ignore


def sample_fachabteilungen(count: int = 100) -> np.ndarray:
    """Funktion zum Erzeugen zufälliger Fachabteilungen.

    Hinweise: Bisher wird nur ein Teil der realen Fachabteilungen abgebildet.

    Args:
        count (int, optional): Anzahl der zu erzeugenden Fachabteilungen. Default ist 100.

    Returns:
        np.ndarray: Array mit allen erzeugten Fachabteilungen.
    """
    if not count > 0:
        raise AssertionError
    possible_fachabteilungen = [
        "Anästhesiologie und Intensivmedizin",
        "Innere Medizin",
        "Orthopädie",
        "Angiologie",
        "Pneumologie",
        "Kardiologie",
        "Gastroenterologie",
        "Neurologie",
        "Hepatologie",
        "Urologie",
        "Radiologie und Nuklearmedizin",
        "Hämatologie",
        "Nephrologie",
        "Frauenheilkunde und Geburtshilfe",
        "Pädiatrie",
        "Kinderonkologie",
        "Geriatrie",
        "HNO – Hals-Nasen-Ohren-Fachabteilung",
        "Zahn-, Mund- und Kieferheilkunde",
        "Onkologie",
        "Palliativstation",
    ]
    return np.random.choice(possible_fachabteilungen, count)


def sample_geschlechter(count: int = 100) -> np.ndarray:
    """Funktion zum Erzeugen zufälliger Geschlechtsangaben.

    Hinweise: Bisher werden nur Männer und Frauen berücksichtigt.

    Args:
        count (int, optional): Anzahl der zu erzeugenden Geschlechtsangaben. Default ist 100.

    Returns:
        np.ndarray: Array mit allen erzeugten Geschlechtsangaben.
    """
    if not count > 0:
        raise AssertionError
    possible_geschlechter = ["männlich", "weiblich"]
    return np.random.choice(possible_geschlechter, count, p=[0.5, 0.5])


def sample_kh_aufnahmeart(count: int = 100) -> np.ndarray:
    """Funktion zum erzeugen zufälliger Aufnahmearten.

    Hinweis: Es wird zwischen Normalfall und Notfall unterschieden.

    Args:
        count (int, optional): Anzahl der zu erzeugenden Aufnahmearten. Default ist 100.

    Returns:
        np.ndarray: Array mit allen erzeugten Aufnahmearten.
    """
    if not count > 0:
        raise AssertionError
    possible_aufnahmearten = ["Normalfall", "Notfall"]
    return np.random.choice(possible_aufnahmearten, count, p=[0.8, 0.2])


def sample_icds(count: int = 100) -> typing.List[str]:
    """Funktion zum erzeugen zufälliger ICD Codes.

    Args:
        count (int, optional): Anazhl der zu erzeugenden ICD Codes. Default ist 100.

    Returns:
        typing.List[str]: Liste mit allen erzeugten ICD Codes.
    """
    if not count > 0:
        raise AssertionError
    # Create random letter:
    letters = string.ascii_uppercase
    rand_letters = random.choices(letters, k=count)
    # Create random two-digit-number
    ids = np.random.choice(range(100), count)
    subids = np.random.choice(range(100), count)
    numbers = [str(id_ % 100).zfill(2) for id_ in ids]
    subnumbers = [str(id_ % 100) for id_ in subids]
    complete_icds = [
        rand_letters[i] + numbers[i] + "." + subnumbers[i] for i in range(count)
    ]
    return complete_icds


def sample_heilmittel_positionsnummern(count: int = 100) -> typing.List[str]:
    """Funktion zum erzeugen zufälliger Heilmittel-Positionsnummern.

    Args:
        count (int, optional): Anzahl der zu erzeugenden Heilmittel-Positionsnummern. Default \
        ist 100.

    Returns:
        typing.List[str]: Liste mit allen erzeugten Heilmittel-Positionsnummern.
    """
    if not count > 0:
        raise AssertionError
    ids = random.choices(range(100000), k=count)
    numbers = [str(id_ % 100000).zfill(5) for id_ in ids]
    return numbers


def sample_hilfsmittelnummern(count: int = 100) -> typing.List[str]:
    """Funktion zum Erzeugen zufälliger Hilfsmittelnummern.

    Args:
        count (int, optional): Anzahl der zu erzeugenden Hilfsmittelnummern. Default ist 100.

    Returns:
        typing.List[str]: Liste mit allen erzeugten Hilfsmittelnummern.
    """
    if not count > 0:
        raise AssertionError
    ids = random.choices(range(10000000000), k=count)
    numbers = [str(id_ % 10000000000).zfill(10) for id_ in ids]
    return numbers


def sample_atcs(count: int = 100) -> typing.List[str]:
    """Funktion zum Erzeugen zufälliger ATCs.

    Hinweis: ATC steht für Anatomisch-Therapeutisch-Chemisches Klassifikationssystem.
    https://de.wikipedia.org/wiki/Anatomisch-Therapeutisch-Chemisches_Klassifikationssystem

    Args:
        count (int, optional): Anzahl der zu erzeugenden ATCs. Default ist 100.

    Returns:
        typing.List[str]: Liste mit allen erzeugten ATCs.
    """
    if not count > 0:
        raise AssertionError
    letters = string.ascii_uppercase
    rand_letter_1 = random.choices(letters, k=count)
    rand_letter_2 = random.choices(letters, k=count)
    rand_letter_3 = random.choices(letters, k=count)
    # Create random two-digit-number
    ids = random.choices(range(100), k=count)
    subids = random.choices(range(100), k=count)
    numbers = [str(id_ % 100).zfill(2) for id_ in ids]
    subnumbers = [str(id_ % 100).zfill(2) for id_ in subids]
    complete_atcs = [
        rand_letter_1[i]
        + numbers[i]
        + rand_letter_2[i]
        + rand_letter_3[i]
        + subnumbers[i]
        for i in range(count)
    ]
    return complete_atcs


def generate_dataframe_arzneimittel(
    unique_kvnrs: typing.List[str], count: int = 1000
) -> pd.DataFrame:
    """Funktion zum Erzeugen zufälliger Arzneimitteldaten.

    Aus der Liste an KVNRs werden zufällige KVNRs gezogen. Die Anzahl liegt dabei \
    bei der Anzahl an gewünschten Arzneimitteldaten (doppelte KVNRs können hierbei \
    vorkommen, default ist 1000). Anschließend werden zufällige ATCs und \
    Datumsangaben kreiert und als Dataframe zurück gegeben.

    Args:
        unique_kvnrs (typing.List[str]): Liste mit KVNRs.
        count (int, optional): Anzahl der zu erzeugenden Arzneimitteldaten. Default ist 1000.

    Returns:
        pd.DataFrame: Arzneimitteldaten die die KVNR, ATC, Datum und Einspielungsdaten beinhalten.
    """
    if not count > 0:
        raise AssertionError
    kvnrs = np.random.choice(unique_kvnrs, size=count, replace=True).tolist()
    atcs = sample_atcs(count=count)
    dates = sample_random_dates(count=count)
    return pd.DataFrame(
        {"KVNR": kvnrs, "ATC": atcs, "Datum": dates, "Eingespielt": dates}
    )


def generate_dataframe_stammdaten(count: int = 100) -> pd.DataFrame:
    """Funktion zum erzeugen zufälliger Stammdaten.

    Die Stammdaten enthalten eine KVNR, Angaben zum Geschlecht, ein Geburts- sowie Todesdatum.
    Hinweis: Es ist möglich, dass diese Daten teilweise semantisch inkorrekt sind. D.h.
    TODO

    Args:
        count (int, optional): Anazhl der zu erzegenden Stammdaten. Default ist 100.

    Returns:
        pd.DataFrame: Ein Dataframe aller erzeugter Stammdaten.
    """
    if not count > 0:
        raise AssertionError

    ids = sample_unique_kvnrs(count)

    # Now generate Geschlechter
    geschlecht = sample_geschlechter(count)

    # Generate Geburtstag
    birth_dates = sample_random_dates(
        count=count,
    )

    life = pd.DataFrame(data=birth_dates, columns=["Datum"])
    life_length = pd.to_timedelta(
        np.random.randint(365, 100 * 365, size=len(birth_dates)), unit="d"
    )
    life["Tod"] = life["Datum"] + life_length
    life["Tod"][life["Tod"] > pd.to_datetime(datetime.datetime.now())] = pd.NaT
    return pd.DataFrame(
        {
            "KVNR": ids,
            "Geschlecht": geschlecht,
            "Geburtstag": birth_dates,
            "Todestag": life["Tod"],
        }
    )


def generate_21c_dataframe_stammdaten(count: int = 100) -> pd.DataFrame:
    """Funktion zum erzeugen zufälliger Stammdaten.

    Die Stammdaten enthalten eine Dataframe einer natürlichen Person, die eine OID, \
    einen Vor- und Nachnamen, Angaben zum Geschlecht und ein Geburtsdatum enthalten.
    Das zweite Dataframe enthält Angaben zum Ordnungbegriff, die eine OID, Typ, Wert \
    und Ordnungsbegriffstatus enthahalten.
    Hinweis: Es ist möglich, dass diese Daten teilweise semantisch inkorrekt sind. D.h.
    TODO

    Args:
        count (int, optional): Anzahl der zu erzegenden Stammdaten. Default ist 100.

    Returns:
        Ein Tuple mit Dataframes, das in gegebender Reihenfolge Angaben zu einer \
        natürlichen Person und einem Ordnungsbegriff ausgibt.
    """
    if not count > 0:
        raise AssertionError

    ids = sample_unique_kvnrs(count)

    # Now generate Geschlechter
    geschlecht = np.random.choice([1, 2], count)

    # Generate Geburtstag
    birth_dates = sample_random_dates(
        count=count,
    )

    # oids
    oids = random.sample(range(26000000000), count)

    # vornamen
    vornamen = [fake.first_name() for i in range(count)]

    # namen
    names = [fake.last_name() for i in range(count)]

    nat_person = pd.DataFrame(
        {
            "OID": oids,
            "VORNAME": vornamen,
            "NAME": names,
            "GESCHLECHT": geschlecht,
            "GEBURTSDATUM": birth_dates,
        }
    )
    ordnungsbegriff = pd.DataFrame(
        {
            "PARTNER_PARTNER_ID": oids,
            "TYP": [9] * count,
            "WERT": ids,
            "ORDNUNGSBEGRIFFSTATUS": [1] * count,
        }
    )
    return nat_person, ordnungsbegriff


def generate_dataframe_versicherungszeiten(
    unique_kvnrs: typing.List[str],
) -> pd.DataFrame:
    """Generate Versicherungszeiten-Dataframe

    Args:
        unique_kvnrs (typing.List[str]): List of unique KVNRs

    Returns:
        pd.DataFrame: Versicherungszeiten-Dataframe
    """
    res = pd.DataFrame(
        {
            "KVNR": [],
            "beginn": [],
            "ende": [],
        }
    )
    for i in unique_kvnrs:
        # TODO: Sample varying number of records per insured person
        res_datum = (
            sample_random_dates(
                count=2 * 5,
                start=pd.to_datetime("1930-01-01"),
                end=pd.to_datetime(datetime.datetime.now()),
            )
            .sort_values()
            .to_frame()
            .values
        )
        if random.random() >= 0.5:
            res_datum[9] = pd.to_datetime("2200-01-01")
        tmp = pd.DataFrame(
            {
                "KVNR": [i] * (len(res_datum) // 2),
                "beginn": [entry[0] for entry in res_datum[::2]],
                "ende": [entry[0] for entry in res_datum[1::2]],
            }
        )
        res = pd.concat([res, tmp], ignore_index=True)

    return res


def generate_dataframe_krankenhausaufenthalt(
    unique_kvnrs: typing.List[str], count: int = 100
) -> pd.DataFrame:
    """Generate Krankenhausaufenthalts-Dataframe

    Args:
        unique_kvnrs (typing.List[str]): List of unique KVNRs
        count (int, optional): Count of entries. Defaults to 100.

    Returns:
        pd.DataFrame: Krankenhausaufenthalts-Dataframe
    """
    if not count > 0:
        raise AssertionError
    kvnrs = np.random.choice(unique_kvnrs, size=count, replace=True).tolist()
    icds = sample_icds(count=count)
    kh_von = sample_random_dates(count=count)
    art = sample_kh_aufnahmeart(count=count)
    kh_bis = kh_von + pd.to_timedelta(np.random.choice(range(30), count), "D")
    kh_fachabteilungen = sample_fachabteilungen(count=count)
    return pd.DataFrame(
        {
            "KVNR": kvnrs,
            "ICD": icds,
            "kh_von": kh_von,
            "kh_bis": kh_bis,
            "Fachabteilung": kh_fachabteilungen,
            "aufnahmeart": art,
        }
    )


def generate_dataframe_arztkontakte(
    unique_kvnrs: typing.List[str], count: int = 100
) -> pd.DataFrame:
    """Generate Arztkontakte-Dataframe

    Args:
        unique_kvnrs (typing.List[str]): List of unique KVNRs
        count (int, optional): Count of entries. Defaults to 100.

    Returns:
        pd.DataFrame: Arztkontakte-Dataframe
    """
    if not count > 0:
        raise AssertionError
    kvnrs = np.random.choice(unique_kvnrs, size=count, replace=True).tolist()
    icds = sample_icds(count=count)
    arztbesuch_am = sample_random_dates(count=count)
    return pd.DataFrame(
        {
            "KVNR": kvnrs,
            "ICD": icds,
            "Arztbesuch_am": arztbesuch_am,
            "Arztbesuch_bis": arztbesuch_am,
        }
    )


def generate_dataframe_hilfsmittel(
    unique_kvnrs: typing.List[str], count: int = 100
) -> pd.DataFrame:
    """Generate Hilfsmittel-Dataframe

    Args:
        unique_kvnrs (typing.List[str]): List of unique KVNRs
        count (int, optional): Count of entries. Defaults to 100.

    Returns:
        pd.DataFrame: Hilfsmittel-Dataframe
    """
    if not count > 0:
        raise AssertionError
    kvnrs = np.random.choice(unique_kvnrs, size=count).tolist()
    himis = sample_hilfsmittelnummern(count=count)
    lm_von = sample_random_dates(count=count)
    return pd.DataFrame(
        {
            "KVNR": kvnrs,
            "Hilfsmittelnummer": himis,
            "lm_von": lm_von,
            "lm_bis": lm_von,
        }
    )


def generate_dataframe_heilmittel(
    unique_kvnrs: typing.List[str], count: int = 100
) -> pd.DataFrame:
    """Generate Heilmittel-Dataframe

    Args:
        unique_kvnrs (typing.List[str]): List of unique KVNRs
        count (int, optional): Count of entries. Defaults to 100.

    Returns:
        pd.DataFrame: Heilmittel-Dataframe
    """
    if not count > 0:
        raise AssertionError
    kvnrs = np.random.choice(unique_kvnrs, size=count, replace=True).tolist()
    heilmittel = sample_heilmittel_positionsnummern(count=count)
    dates = sample_random_dates(count=count)
    return pd.DataFrame(
        {
            "KVNR": kvnrs,
            "Positionsnummer": heilmittel,
            "lm_von": dates,
            "lm_bis": dates,
        }
    )


def generate_21c_tables(
    anzahl_versicherte: int = 100, conn=sqlalchemy.engine.base.Engine
) -> None:
    """

    Args:
        anzahl_versicherte (int, optional): _description_. Defaults to 100.
        conn (_type_, optional): _description_. Defaults to sqlalchemy.engine.base.Engine.
    """
    nat_pers, ordnungsbegriff = generate_21c_dataframe_stammdaten(
        count=anzahl_versicherte
    )
    connection_session = conn.connect()
    nat_pers.to_sql("PP_NATUERLICHEPERSON", connection_session, if_exists="replace")
    ordnungsbegriff.to_sql(
        "PP_ORDNUNGSBEGRIFF", connection_session, if_exists="replace"
    )


def generate_all_tables(
    anzahl_versicherte: int = 100, conn=sqlalchemy.engine.base.Engine
) -> None:
    """
    Generiert alle Tabellen

    :param anzahl_versicherte: Anzahl an Versicherten, defaults to 100
    :type anzahl_versicherte: int, optional
    """
    stammdaten = generate_dataframe_stammdaten(count=anzahl_versicherte)
    arzneimittel = generate_dataframe_arzneimittel(
        unique_kvnrs=stammdaten["KVNR"].tolist(), count=10 * anzahl_versicherte
    )
    krankenhaus = generate_dataframe_krankenhausaufenthalt(
        unique_kvnrs=stammdaten["KVNR"].tolist(), count=10 * anzahl_versicherte
    )
    himis = generate_dataframe_hilfsmittel(
        unique_kvnrs=stammdaten["KVNR"].tolist(), count=10 * anzahl_versicherte
    )
    heilmittel = generate_dataframe_heilmittel(
        unique_kvnrs=stammdaten["KVNR"].tolist(), count=10 * anzahl_versicherte
    )
    arztkontakte = generate_dataframe_arztkontakte(
        unique_kvnrs=stammdaten["KVNR"].tolist(), count=10 * anzahl_versicherte
    )
    versicherungsverlauf = generate_dataframe_versicherungszeiten(
        unique_kvnrs=stammdaten["KVNR"].tolist()
    )

    connection_session = conn.connect()
    stammdaten.to_sql("stammdaten", connection_session, if_exists="replace")
    versicherungsverlauf.to_sql(
        "versicherungsverlauf", connection_session, if_exists="replace"
    )
    arzneimittel.to_sql("arzneimittel", connection_session, if_exists="replace")
    krankenhaus.to_sql("kh", connection_session, if_exists="replace")
    himis.to_sql("hilfsmittel", connection_session, if_exists="replace")
    heilmittel.to_sql("heilmittel", connection_session, if_exists="replace")
    arztkontakte.to_sql("arztkontakte", connection_session, if_exists="replace")
