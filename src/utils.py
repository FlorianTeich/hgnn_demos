import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import kuzu
import pathlib
from multiprocessing import cpu_count
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch, Dataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import (MLP, BatchNorm, LayerNorm, GraphConv, MultiAggregation,
                                    SAGEConv, to_hetero)
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric.transforms as T


model_name='all-MiniLM-L6-v2'
model_string_encoder = SentenceTransformer(model_name)

@torch.no_grad()
def encode_strings(df):
    x = model_string_encoder.encode(df.values, show_progress_bar=True)
    return x

def get_loaders(db, LOADER_BATCH_SIZE, train_inds, test_inds):
    feature_store, graph_store = db.get_torch_geometric_remote_backend(4)

    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()
                        ])

    train_loader = NeighborLoader(
        data=(feature_store, graph_store),
        num_neighbors={('diagnosis', 'assigned_to', 'person'): [-1],
                        ('drug', 'consumed_by', 'person'): [-1],
                    },
        batch_size=LOADER_BATCH_SIZE,
        input_nodes=('person', train_inds),
        transform=transform,
        disjoint=True,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        filter_per_worker=False,
    )

    test_loader = NeighborLoader(
        data=(feature_store, graph_store),
        num_neighbors={('diagnosis', 'assigned_to', 'person'): [-1],
                        ('drug', 'consumed_by', 'person'): [-1],
                    },
        batch_size=LOADER_BATCH_SIZE,
        input_nodes=('person', test_inds),
        transform=transform,
        disjoint=True,
        num_workers=4,
        pin_memory=True,
        filter_per_worker=False,
    )
    return train_loader, test_loader


class GraphLevelGNN(torch.nn.Module):
    def __init__(self, hidden_size=32, num_classes=2):
        super().__init__()
        self.conv1 = SAGEConv(-1, hidden_size)
        self.bn1 = BatchNorm(hidden_size)
        self.pool = MultiAggregation(
            aggrs=['mean', 'min', 'max'],
            mode="cat"
            )
        self.lin = Linear(hidden_size*3, num_classes)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor=None) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.pool(x, batch)
        x = self.lin(x)
        return x


def get_model_v1(device, batch):
    learn_rate = 0.01
    aggr = "max"

    batch = batch.to(device)
    metadata = batch.metadata()

    model = GraphLevelGNN()
    model = to_hetero(model, metadata, aggr=aggr, debug=True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
    return model, optimizer


def create_kuzu_node_table(conn, tablename="paper", feature_dim=128):
    conn.execute(
        "CREATE NODE TABLE " + tablename + "(id INT64, x FLOAT[" + str(feature_dim) + "], y FLOAT, "
        "PRIMARY KEY (id));")

def copy_nodes_to_kuzu(conn,
                       tablename="paper",
                       ids_path="ids.npy",
                       node_feature_path="node_feature.npy",
                       node_label_path="node_label.npy"):
    conn.execute('COPY ' + tablename + ' FROM ("%s",  "%s", "%s") BY COLUMN;' %
             (ids_path, node_feature_path, node_label_path))

def create_kuzu_relationship_table(conn, relationshipname="cites", from_="papers", to_="papers"):
    conn.execute("CREATE REL TABLE " + relationshipname +
                 "(FROM " + from_ + " TO " + to_ + ", MANY_MANY);")

def copy_edges_to_kuzu(conn, relationshipname="cites", edge_file="edge_index.csv"):
    conn.execute('COPY ' + relationshipname + ' FROM "%s";' % (edge_file))

def kuzu_node_table_from_arrays(conn, tablename="paper", feats=None, labels=None,
                    path=str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/"):
    # Create csvs and npy files from df
    ids_path = path + tablename + "_ids.npy"
    node_feature_path = path + tablename + "_node_features.npy"
    node_label_path = path + tablename + "_node_labels.npy"
    ids = np.arange(feats.shape[0])
    np.save(ids_path, ids)

    with open(node_feature_path, 'wb') as f:
        np.save(f, np.ascontiguousarray(feats).astype(np.float32))

    with open(node_label_path, 'wb') as f:
        np.save(f, np.ascontiguousarray(labels).astype(np.float32))

    # Create Table
    create_kuzu_node_table(conn, tablename=tablename, feature_dim=feats.shape[1])

    # Fill Table
    copy_nodes_to_kuzu(conn, tablename=tablename,
                       ids_path=ids_path,
                       node_feature_path=node_feature_path,
                       node_label_path=node_label_path)

def kuzu_edges_from_tensor(conn, edge_index, relationshipname, from_, to_,
                           path=str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/"):
    # Create csv file
    csv_path = path + relationshipname + "_edge_index.csv"
    csvfile = open(csv_path, 'w')
    for i in tqdm(range(edge_index.shape[1])):
        csvfile.write(str(edge_index[0, i]) + ',' + str(edge_index[1, i]) + '\n')
    csvfile.close()

    # create relationship in kuzu
    create_kuzu_relationship_table(conn, relationshipname=relationshipname,
                                   from_=from_, to_=to_)

    # Copy edges
    copy_edges_to_kuzu(conn, relationshipname=relationshipname, edge_file=csv_path)