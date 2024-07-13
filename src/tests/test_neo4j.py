import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from scripts.utils_neo4j import (
    clean_db,
    conn,
    create_index,
    create_single_node,
    create_two_nodes_and_a_relation,
    get_all_indexes,
    get_db_query_result_as_networkx,
    match_two_node_types_and_relation,
    return_all_nodes,
)


def execute_in_clean_db(func):
    def wrapper_func():
        clean_db()
        func()
        clean_db()

    return wrapper_func


@execute_in_clean_db
def test_neo4j_connection():
    assert conn is not None


@execute_in_clean_db
def test_neo4j_clean_db():
    response = return_all_nodes()
    assert len(response) == 0


@execute_in_clean_db
def test_neo4j_create_node():
    response = create_single_node("Person", {"name": "John Doe"})
    assert response[0]["n"]["name"] == "John Doe"


@execute_in_clean_db
def test_neo4j_create_relation():
    type1: str = "Person"
    properties1: dict = {"name": "John Doe"}
    type2: str = "Email"
    properties2: dict = {"email": "[email protected]"}
    relation: str = "HAS_EMAIL"

    response = create_two_nodes_and_a_relation(
        type1=type1,
        properties1=properties1,
        type2=type2,
        properties2=properties2,
        relation=relation,
    )
    response = match_two_node_types_and_relation(
        type1=type1, type2=type2, relation=relation
    )
    assert response[0]["entity_type_1"]["name"] == "John Doe"
    assert response[0]["entity_type_2"]["email"] == "[email protected]"
    assert response[0]["relation"].type == "HAS_EMAIL"
    assert response[0]["relation"].start_node["name"] == "John Doe"
    assert response[0]["relation"].end_node["email"] == "[email protected]"


@execute_in_clean_db
def test_neo4j_create_index():
    create_index("Person", "name", "index_01")
    response = get_all_indexes()
    assert response[0]["name"] == "index_01"
    assert response[0]["properties"] == ["name"]


@execute_in_clean_db
def test_get_db_query_result_as_networkx():
    type1: str = "Person"
    properties1: dict = {"name": "John Doe"}
    type2: str = "Email"
    properties2: dict = {"email": "[email protected]"}
    relation: str = "HAS_EMAIL"

    create_two_nodes_and_a_relation(
        type1=type1,
        properties1=properties1,
        type2=type2,
        properties2=properties2,
        relation=relation,
    )
    graph_nx = get_db_query_result_as_networkx(
        f"""MATCH (e:{type1})
        WITH e
        LIMIT 1
        MATCH (e)-[r]-(relatedNodes)
        RETURN e, r, relatedNodes"""
    )
    assert len(graph_nx.nodes) == 2
    assert len(graph_nx.edges) == 1
