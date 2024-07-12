import os
import time
import torch
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv("../.env")

NEO4J_USER = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_HOST = "bolt://127.0.0.1:7687"


class Neo4jConnection:
    def __init__(self, uri: str, user: str, pwd: str):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(
                self.__uri, auth=(self.__user, self.__pwd)
            )
        except Exception as e:
            print("Failed to create the driver:", e)

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query: str, parameters=None, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = (
                self.__driver.session(database=db)
                if db is not None
                else self.__driver.session()
            )
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response


conn = Neo4jConnection(uri=NEO4J_HOST, user=NEO4J_USER, pwd=NEO4J_PASSWORD)

driver = GraphDatabase.driver(NEO4J_HOST, auth=(NEO4J_USER, NEO4J_PASSWORD))


def fetch_data(query, params: dict = {}) -> pd.DataFrame:
    """Fill query with data

    Args:
        params (dict): Parameters to fill the query

    Returns:
        pd.DataFrame: Dataframe with the query results"""
    with driver.session() as session:
        result = session.run(query, params)
        return pd.DataFrame([r.values() for r in result], columns=result.keys())


def load_node(cypher, index_col, encoders=None, **kwargs):
    """Execute the cypher query and retrieve data from Neo4j"""
    df = fetch_data(cypher)
    df.set_index(index_col, inplace=True)
    # Define node mapping
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    # Define node features
    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)
    return x, mapping


def load_edge(
    cypher: str,
    src_index_col: str,
    src_mapping: str,
    dst_index_col: str,
    dst_mapping: str,
    encoders=None,
    **kwargs,
):
    """Execute the cypher query and retrieve data from Neo4j"""
    df = fetch_data(cypher)
    # Define edge index
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])
    # Define edge features
    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


def insert_data(query, rows: pd.DataFrame, batch_size: int = 10_000) -> dict:
    """Function to handle the updating the Neo4j database in batch mode.

    Args:
        query (str): Cypher query to execute
        rows (pd.DataFrame): Dataframe with the data to insert
        batch_size (int): Size of the batch

    Returns:
        dict: Dictionary with the total number of rows inserted, number of batches and
        time taken to insert the data
    """
    total = 0
    batch = 0
    start = time.time()
    result = None

    while batch * batch_size < len(rows):

        res = conn.query(
            query,
            parameters={
                "rows": rows[batch * batch_size : (batch + 1) * batch_size].to_dict(
                    "records"
                )
            },
        )
        total += res[0]["total"]
        batch += 1
        result = {"total": total, "batches": batch, "time": time.time() - start}
        print(result)

    return result
