"""Manage Graph Datasets and prepare them for GNNs"""
from tqdm import trange, tqdm
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from typing import Optional
import logging
import pandas as pd
import numpy as np
import typer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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
    df = pd.DataFrame(
        {
            "entity_id": np.arange(num_samples),
            "entity_name": [f"entity_{i}" for i in range(num_samples)],
            "feature_1": np.random.rand(num_samples),
            "feature_2": np.random.rand(num_samples),
        }
    )

    # Create related entity table
    df_rel = pd.DataFrame(
        {
            "entity_id": np.arange(num_related_entities),
            "related_entity_id": np.random.choice(
                df["entity_id"], num_related_entities
            ),
            "feature_1": np.random.rand(num_related_entities),
            "feature_2": np.random.rand(num_related_entities),
            "feature_3": np.random.rand(num_related_entities),
        }
    )

    return df, df_rel


def store_data_in_relational_db(
    df: pd.DataFrame,
    df_rel: pd.DataFrame,
    db: str = "sqlite",
    args: Optional[dict] = None,
) -> None:
    """Store data in relational database

    Args:
        df (pd.DataFrame): Main entity table
        df_rel (pd.DataFrame): Related entity table
        db (str): Database to store data
        args (Optional[dict]): Additional arguments

    Returns:
        None
    """
    log.info("✨ Storing data in relational database")
    # Store data in relational database
    if db == "sqlite":
        # Store both tables in a SQLite database and use foreign key constraints and relationships

        # Create SQLite database
        engine = create_engine("sqlite:///data.db")
        Base = declarative_base()

        # Define main entity table
        class Entity(Base):
            __tablename__ = "entity"
            entity_id = Column(Integer, primary_key=True)
            entity_name = Column(String)
            feature_1 = Column(Float)
            feature_2 = Column(Float)
            related_entities = relationship("RelatedEntity")

        # Define related entity table
        class RelatedEntity(Base):
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
        for i, row in tqdm(df.iterrows()):
            entity = Entity(
                entity_id=row["entity_id"],
                entity_name=row["entity_name"],
                feature_1=row["feature_1"],
                feature_2=row["feature_2"],
            )
            session.add(entity)
        session.commit()

        for i, row in tqdm(df_rel.iterrows()):
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
        raise ValueError(f"Database {db} not supported")


@app.command()
def generate_toy_data_in_relational_db(
    num_samples: int = 1_000,
    num_related_entities: int = 10_000,
    relatinal_db: str = "sqlite",
) -> None:
    """Generate toy data

    Args:
        num_samples (int): Number of samples to generate

    Returns:
        None
    """
    args: Optional[dict] = None
    df, df_rel = generate_toy_data(num_samples, num_related_entities)
    store_data_in_relational_db(df, df_rel, db=relatinal_db, args=args)
    log.info("🚀 Data generated and stored in relational database")


@app.command()
def convert_relational_db_to_graph_db() -> None:
    """Convert relational database to graph database"""
    log.info("🔁 Converting relational database to graph database")
    # Convert relational database to graph database
    pass


if __name__ == "__main__":
    app()
