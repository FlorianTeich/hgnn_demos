import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import Linear, MultiAggregation, SAGEConv


def edges_diagnosis():
    persons = pd.read_parquet("./data/persons.parquet")
    diagnosis_data = pd.read_parquet("./data/diagnoses.parquet")
    return np.flip(
        persons.merge(diagnosis_data, on="PID", how="inner")[
            ["index_x", "index_y"]
        ].values.T
    )


def edges_drugs():
    persons = pd.read_parquet("./data/persons.parquet")
    drugs_data = pd.read_parquet("./data/drugs.parquet")
    return np.flip(
        persons.merge(drugs_data, on="PID", how="inner")[
            ["index_x", "index_y"]
        ].values.T
    )


class GraphLevelGNN(torch.nn.Module):
    def __init__(self, hidden_size=32, num_classes=2):
        super().__init__()
        self.conv1 = SAGEConv(-1, hidden_size)
        self.pool = MultiAggregation(aggrs=["mean", "min", "max"], mode="cat")
        self.lin = Linear(hidden_size * 3, num_classes)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor = None) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.pool(x, batch)
        x = self.lin(x)
        return x
