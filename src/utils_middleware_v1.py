import numpy
import torch

from utils_neo4j import *


def add_versicherte_v1(versicherte):
    """Add Versicherter"""
    # Create indexes for the properties used in the MERGE statement
    conn.query("CREATE INDEX FOR (v:versicherter) ON (v.KVNR)")
    conn.query("CREATE INDEX FOR (v:versicherter) ON (v.alter)")
    conn.query("CREATE INDEX FOR (v:versicherter) ON (v.geschlecht)")
    conn.query("CREATE INDEX FOR (v:versicherter) ON (v.label)")

    query = """
            UNWIND $rows AS row
            MERGE (v:versicherter {KVNR: row.KVNR, alter: row.alter, geschlecht: row.Geschlecht, label: row.label}) ON CREATE SET v.id = row.KVNR
            RETURN count(*) as total
            """
    return conn.query(query, parameters={"rows": versicherte.to_dict("records")})


def add_diagnosen_v1(rows, batch_size=10000):
    """Add Diagnose"""
    # Create indexes for the properties used in the MERGE statement
    conn.query("CREATE INDEX FOR (d:diagnose) ON (d.name)")
    conn.query("CREATE INDEX FOR (d:diagnose) ON (d.icd_age)")
    conn.query("CREATE INDEX FOR (d:diagnose) ON (d.icd_short)")

    query = """
            UNWIND $rows AS row
            MERGE (:diagnose {name: row.ICD, icd_age: row.icd_age, icd_short: row.icd_short})
            RETURN count(*) as total
            """
    return insert_data(query, rows, batch_size)


def add_relations_v1(rows, batch_size=5000):
    """
    Adds versicherten nodes and diagnose relationships to the Neo4j graph as a
    batch job
    """

    # Create indexes for the properties used in the MERGE statement
    conn.query("CREATE INDEX versicherter_id FOR (v:versicherter) ON (v.id)")
    conn.query("CREATE INDEX diagnose_name FOR (d:diagnose) ON (d.name)")

    query = """
    UNWIND $rows as row
    MERGE (v:versicherter {id:row.KVNR})

    // connect versicherte
    WITH v, row.ICD AS icds
    MATCH (d:diagnose {name: icds})
    MERGE (v)-[:DIAGNOSTIZIERT]->(d)

    RETURN count(distinct d) as total
    """

    return insert_data(query, rows, batch_size)


class SequenceEncoder(object):
    """The 'SequenceEncoder' encodes raw column strings into embeddings."""

    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(
            df.values,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device,
        )
        return x.cpu()


class SequenceEncoderCached(object):
    """The 'SequenceEncoder' encodes raw column strings into embeddings."""

    def __init__(self, cache, mapping):
        self.cache = cache
        self.mapping = mapping

    @torch.no_grad()
    def __call__(self, df):
        curr = []
        for entry in df.values:
            curr.append(self.cache[self.mapping[entry]])
        return torch.stack(curr)


class IdentityEncoder(object):
    """The 'IdentityEncoder' takes the raw column values and converts them to
    PyTorch tensors."""

    def __init__(self, dtype=None, is_list=False):
        self.dtype = dtype
        self.is_list = is_list

    def __call__(self, df):
        if self.is_list:
            return torch.stack([torch.tensor(el) for el in df.values])
        return torch.from_numpy(df.values).to(self.dtype)


class ScalarIdentityEncoder(object):
    """
    The 'ScalarIdentityEncoder' takes the raw column values and converts them to
    PyTorch tensors."""

    def __init__(self, dtype=None, is_list=False):
        self.dtype = dtype
        self.is_list = is_list

    def __call__(self, df):
        if self.is_list:
            return torch.stack([torch.tensor(el) for el in df.values])
        return torch.unsqueeze(torch.from_numpy(df.values).to(self.dtype), -1)


class IcdEncoder(object):
    """
    The 'ICDEncoder' converts Single-Characters to a class.
    """

    def __init__(self):
        pass

    def __call__(self, df):
        x = torch.zeros(len(df), 26)
        for i, col in enumerate(df.values):
            x[i, ord(col) - 97] = 1
        # x = torch.hstack([torch.sum(
        #    x[:, :4], axis=1, keepdims=True),
        #    torch.sum(x[:, 4:], axis=1, keepdims=True)])
        return x
