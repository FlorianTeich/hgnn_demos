import os
import time

import pandas as pd
import torch
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


def clean_db():
    # Remove nodes and relationships
    query = "MATCH (n) DETACH DELETE n"
    conn.query(query)
    # Remove indexes
    query = "SHOW INDEXES"
    response = conn.query(query)
    for index in response:
        query = f"DROP INDEX {index['name']} IF EXISTS"
        conn.query(query)
    # Remove constraints
    query = "SHOW CONSTRAINTS"
    response = conn.query(query)
    for constraint in response:
        query = f"DROP CONSTRAINT {constraint['name']}_unique IF EXISTS"
        conn.query(query)


def return_all_nodes():
    query = "MATCH (n) RETURN n"
    return conn.query(query)


def create_single_node(label: str, properties: dict):
    """Create a single node in Neo4j

    Args:
        label (str): Label of the node
        properties (dict): Properties of the node

    Returns:
        None
    """
    properties = ", ".join([f"{key}: '{value}'" for key, value in properties.items()])
    query = f"CREATE (n:{label} {{{properties}}}) RETURN n"
    return conn.query(query)


def get_all_indexes():
    query = "SHOW INDEXES"
    return conn.query(query)


def get_all_constraints():
    query = "SHOW CONSTRAINTS"
    return conn.query(query)


def create_two_nodes_and_a_relation(
    type1: str = "Person",
    properties1: dict = {"name": "John Doe"},
    type2: str = "Email",
    properties2: dict = {"email": "[email protected]"},
    relation: str = "HAS_EMAIL",
):
    property_string1 = ", ".join(
        [f"{key}: '{value}'" for key, value in properties1.items()]
    )
    property_string2 = ", ".join(
        [f"{key}: '{value}'" for key, value in properties2.items()]
    )
    query = f"""
    CREATE (entity_type_1:{type1} {{{property_string1}}})-[:{relation}]->(entity_type_2:{type2} {{{property_string2}}})
    """
    return conn.query(query)


def match_two_node_types_and_relation(
    type1: str = "Person", type2: str = "Email", relation: str = "HAS_EMAIL"
):
    query = f"""
    MATCH (entity_type_1:{type1})-[relation:{relation}]->(entity_type_2:{type2}) RETURN entity_type_1, entity_type_2, relation"""
    return conn.query(query)


def create_index(label: str, property: str, indexname: str):
    query = f"""
    CREATE INDEX {indexname} FOR (p:{label}) ON (p.{property})
    """
    return conn.query(query)


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
