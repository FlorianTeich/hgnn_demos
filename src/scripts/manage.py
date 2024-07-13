"""Manage Graph Datasets and prepare them for GNNs"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
import typer
from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    inspect,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from tqdm import tqdm
from utils_neo4j import conn, insert_data

DEFAULT_SQLITE_DB = "sqlite:///data.db"
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
# Set format
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# set format for console
console = logging.StreamHandler()
console.setFormatter(formatter)
log.addHandler(console)
log.propagate = False

app = typer.Typer()


def generate_toy_data(
    num_samples: int = 1_000, num_related_entities: int = 10_000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate toy data

    Args:
        num_samples (int): Number of samples to generate

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple of main entity table and related entity table
    """
    log.info("⭐ Generating data")
    # Create main entity table
    dataframe = pd.DataFrame(
        {
            "entity_id": np.arange(num_samples),
            "entity_name": [f"entity_{i}" for i in range(num_samples)],
            "feature_1": np.random.rand(num_samples),
            "feature_2": np.random.rand(num_samples),
        }
    )

    # Create related entity table
    dataframe_rel = pd.DataFrame(
        {
            "entity_id": np.arange(num_related_entities),
            "related_entity_id": np.random.choice(
                dataframe["entity_id"], num_related_entities
            ),
            "feature_1": np.random.rand(num_related_entities),
            "feature_2": np.random.rand(num_related_entities),
            "feature_3": np.random.rand(num_related_entities),
        }
    )

    return dataframe, dataframe_rel


def store_data_in_relational_db(
    dataframe: pd.DataFrame,
    dataframe_rel: pd.DataFrame,
    database: str = "sqlite",
    args: Optional[dict] = None,
) -> None:
    """Store data in relational database

    Args:
        dataframe (pd.DataFrame): Main entity table
        dataframe_rel (pd.DataFrame): Related entity table
        database (str): Database to store data
        args (Optional[dict]): Additional arguments

    Returns:
        None
    """
    log.info("✨ Storing data in relational database")
    # Store data in relational database
    if database == "sqlite":
        # Store both tables in a SQLite database and use foreign key constraints and relationships

        # Create SQLite database
        engine = create_engine(DEFAULT_SQLITE_DB)
        Base = declarative_base()

        # Define main entity table
        class Entity(Base):
            """Main entity table"""

            __tablename__ = "entity"
            entity_id = Column(Integer, primary_key=True)
            entity_name = Column(String)
            feature_1 = Column(Float)
            feature_2 = Column(Float)
            related_entities = relationship("RelatedEntity")

        # Define related entity table
        class RelatedEntity(Base):
            """Related entity table"""

            __tablename__ = "related_entity"
            entity_id = Column(Integer, primary_key=True)
            related_entity_id = Column(Integer, ForeignKey("entity.entity_id"))
            feature_1 = Column(Float)
            feature_2 = Column(Float)
            feature_3 = Column(Float)

        # Create tables
        Base.metadata.create_all(engine)

        # Store data
        Session = sessionmaker(bind=engine)
        session = Session()
        for _, row in tqdm(dataframe.iterrows()):
            entity = Entity(
                entity_id=row["entity_id"],
                entity_name=row["entity_name"],
                feature_1=row["feature_1"],
                feature_2=row["feature_2"],
            )
            session.add(entity)
        session.commit()

        for _, row in tqdm(dataframe_rel.iterrows()):
            related_entity = RelatedEntity(
                related_entity_id=row["related_entity_id"],
                entity_id=row["entity_id"],
                feature_1=row["feature_1"],
                feature_2=row["feature_2"],
                feature_3=row["feature_3"],
            )
            session.add(related_entity)
        session.commit()
    else:
        raise ValueError(f"Database {database} not supported")


@app.command()
def generate_toy_data_in_relational_db(
    num_samples: int = 1_000,
    num_related_entities: int = 10_000,
    relational_db: str = "sqlite",
) -> None:
    """Generate toy data

    Args:
        num_samples (int): Number of samples to generate

    Returns:
        None
    """
    args: Optional[dict] = None
    dataframe, dataframe_rel = generate_toy_data(num_samples, num_related_entities)
    store_data_in_relational_db(
        dataframe, dataframe_rel, database=relational_db, args=args
    )
    log.info("🚀 Data generated and stored in relational database")


def add_table_to_graph_db(
    tablename: str,
    tabledata: pd.DataFrame,
    primarykey: str | None = None,
) -> None:
    """Add table to graph database

    Args:
        tablename (str): Name of the table
        tabledata (pd.DataFrame): Dataframe of the table
        pk (str | None): Primary key of the table

    Returns:
        None
    """
    log.info("🔵 Adding table %s to graph database", tablename)
    # Add table to graph database
    # conn.query(f"DROP CONSTRAINT {tablename}_unique")
    if primarykey is not None:
        conn.query(
            f"CREATE CONSTRAINT {tablename}_unique IF NOT EXISTS FOR "
            + f"(e:{tablename}) REQUIRE e.{primarykey} IS UNIQUE"
        )

    for column in tabledata.columns:
        # conn.query(f"DROP INDEX {tablename}_{column} IF EXISTS")
        conn.query(
            f"CREATE INDEX {tablename}_{column} FOR (e:{tablename}) ON (e.{column})"
        )

    columnspecs = ", ".join([f"{column}: row.{column}" for column in tabledata.columns])
    query = f"""
            UNWIND $rows AS row
            MERGE (e:{tablename} {{{columnspecs}}}) ON CREATE SET e.{primarykey} = row.{primarykey}
            RETURN count(*) as total
            """
    return conn.query(query, parameters={"rows": tabledata.to_dict("records")})


def add_relation_to_graph_db(
    relation_name: str,
    source_table: str,
    target_table: str,
    source_column: str,
    target_column: str,
    relationdata: pd.DataFrame,
    batch_size: int = 5_000,
) -> None:
    """Add relation to graph database

    Args:
        relation_name (str): Name of the relation
        source_table (str): Source table
        target_table (str): Target table
        source_column (str): Source column
        target_column (str): Target column
        relationdata (pd.DataFrame): Dataframe with the relation data
        batch_size (int): Batch size for the relation data

    Returns:
        None"""
    # Create indexes for the properties used in the MERGE statement
    query = f"""
    UNWIND $rows as row
    MERGE ({source_table}:{source_table} {{{source_column}: row.{source_column}}})

    WITH {source_table}, row.{target_column} AS {target_column}
    MATCH (t:{target_table} {{{target_column}: {target_column}}})
    MERGE (s)-[:{relation_name}]->(t)

    RETURN count(distinct t) as total
    """

    return insert_data(query, relationdata, batch_size)


@app.command()
def convert_relational_db_to_graph_db(source: str = DEFAULT_SQLITE_DB) -> None:
    """Convert relational database to graph database"""
    log.info("🔁 Converting relational database to graph database")
    # Convert relational database to graph database
    # Get list of tables inside the relational database
    engine = create_engine(source)
    inspector = inspect(engine)
    relational_tables = inspector.get_table_names()
    connnection = engine.connect()
    conn.query("MATCH (n) DETACH DELETE n")

    for table in relational_tables:
        # Get pk of a table:
        primarykey = inspector.get_pk_constraint(table)["constrained_columns"][0]
        add_table_to_graph_db(
            table, pd.read_sql(f"SELECT * FROM {table}", connnection), primarykey
        )

    for table in relational_tables:
        relational_relations = inspector.get_foreign_keys(table)
        for relation in relational_relations:
            primarykey = inspector.get_pk_constraint(table)["constrained_columns"][0]
            const_col = relation["constrained_columns"][0]
            add_relation_to_graph_db(
                relation["referred_table"]
                + "_"
                + relation["referred_columns"][0]
                + "_"
                + relation["constrained_columns"][0]
                + "_"
                + table,
                table,
                relation["referred_table"],
                relation["constrained_columns"][0],
                relation["referred_columns"][0],
                pd.read_sql(
                    f"SELECT {primarykey}, {const_col} FROM {table}", connnection
                ),
            )
            # add reverse relationship
            add_relation_to_graph_db(
                table
                + "_"
                + relation["constrained_columns"][0]
                + "_"
                + relation["referred_table"]
                + "_"
                + relation["referred_columns"][0],
                relation["referred_table"],
                table,
                relation["referred_columns"][0],
                relation["constrained_columns"][0],
                pd.read_sql(
                    f"SELECT {const_col}, {primarykey} FROM {table}", connnection
                ),
            )


if __name__ == "__main__":
    # app()
    convert_relational_db_to_graph_db()
