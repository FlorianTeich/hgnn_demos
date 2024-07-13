import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from scripts.utils_neo4j import conn
# import decorator or wrapper for python functions

def execute_in_clean_db(func):
    def wrapper_func():
        clean_db()
        func()
        clean_db()
    return wrapper_func


def clean_db():
    query = "MATCH (n) DETACH DELETE n"
    conn.query(query)

@execute_in_clean_db
def test_neo4j_connection():
    assert conn is not None

@execute_in_clean_db
def test_neo4j_clean_db():
    query = "MATCH (n) RETURN n"
    response = conn.query(query)
    assert len(response) == 0

@execute_in_clean_db
def test_neo4j_create_node():
    query = "CREATE (n:Person {name: 'John Doe'}) RETURN n"
    response = conn.query(query)
    assert response[0]["n"]["name"] == "John Doe"

@execute_in_clean_db
def test_neo4j_fetch_data():
    query = "CREATE (n:Person {name: 'John Doe'}) RETURN n"
    response = conn.query(query)
    assert response[0]["n"]["name"] == "John Doe"

    query = "MATCH (n:Person) RETURN n"
    response = conn.query(query)
    assert response[0]["n"]["name"] == "John Doe"

def test_neo4j_create_relation():
    query = """
    CREATE (n:Person {name: 'John Doe'})-[:HAS_EMAIL]->(e:Email {email: '[email protected]'})
    """
    response = conn.query(query)
    query = "MATCH (n:Person)-[r:HAS_EMAIL]->(e:Email) RETURN n, e, r"
    response = conn.query(query)
    assert response[0]["n"]["name"] == "John Doe"
    assert response[0]["e"]["email"] == "[email protected]"
    assert response[0]["r"].type == "HAS_EMAIL"
    assert response[0]["r"].start_node["name"] == "John Doe"
    assert response[0]["r"].end_node["email"] == "[email protected]"
