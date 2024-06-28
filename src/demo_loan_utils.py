import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import Linear, MultiAggregation, SAGEConv


def edges_trans():
    loan = pd.read_parquet("./data/loan.parquet")
    trans = pd.read_parquet("./data/trans.parquet")
    loan = loan.reset_index()
    trans = trans.reset_index()
    loan["acc"] = loan["account_id"]
    trans["acc"] = trans["account_id"]
    return np.flip(
        loan.merge(trans, on="acc", how="inner")[["index_x", "index_y"]].values.T
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
