import pathlib
from multiprocessing import cpu_count
import yaml
from yaml.loader import SafeLoader
import numpy as np
import torch
import torch_geometric.transforms as T
from sentence_transformers import SentenceTransformer
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
import pandas as pd
import copy

model_name='all-MiniLM-L6-v2'
model_string_encoder = SentenceTransformer(model_name)

@torch.no_grad()
def encode_strings(df):
    x = model_string_encoder.encode(df.values, show_progress_bar=True)
    return x

def load_yaml(path='../default.yml'):
    with open(path) as f:
        data = yaml.load(f, Loader=SafeLoader)
        #print(data)
    return data

def get_loaders(db, LOADER_BATCH_SIZE, train_inds, test_inds, target_entity):
    feature_store, graph_store = db.get_torch_geometric_remote_backend(4)

    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()
                        ])

    train_loader = NeighborLoader(
        data=(feature_store, graph_store),
        num_neighbors=[-1],
        batch_size=LOADER_BATCH_SIZE,
        input_nodes=(target_entity, train_inds),
        transform=transform,
        disjoint=True,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        filter_per_worker=False,
    )

    test_loader = NeighborLoader(
        data=(feature_store, graph_store),
        num_neighbors=[-1],
        batch_size=LOADER_BATCH_SIZE,
        input_nodes=(target_entity, test_inds),
        transform=transform,
        disjoint=True,
        num_workers=4,
        pin_memory=True,
        filter_per_worker=False,
    )
    return train_loader, test_loader


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
                    path="./data/"):
    # Create csvs and npy files from df
    ids_path = path + "/" + tablename + "_ids.npy"
    node_feature_path = path + "/" + tablename + "_node_features.npy"
    node_label_path = path + "/" + tablename + "_node_labels.npy"
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
    csv_path = path + "/" + relationshipname + "_edge_index.csv"
    csvfile = open(csv_path, 'w')
    for i in tqdm(range(edge_index.shape[1])):
        csvfile.write(str(edge_index[0, i]) + ',' + str(edge_index[1, i]) + '\n')
    csvfile.close()

    # create relationship in kuzu
    create_kuzu_relationship_table(conn, relationshipname=relationshipname,
                                   from_=from_, to_=to_)

    # Copy edges
    copy_edges_to_kuzu(conn, relationshipname=relationshipname, edge_file=csv_path)


def auto_enc(df, THRESHOLD_RATIO_CATEGORIES=0.2, THRESHOLD_ABSOLUTE_CATEGORIES=128):
    mapping = dict()
    for i, _ in enumerate(df.columns):
        if df.dtypes.iloc[i] == "float64":
            mapping[str(df.columns[i])] = "proc_raw"
        elif df.dtypes.iloc[i] == "object":
            if (len(df[df.columns[i]].unique()) / len(df) <= THRESHOLD_RATIO_CATEGORIES) & \
                (len(df[df.columns[i]].unique()) <= THRESHOLD_ABSOLUTE_CATEGORIES):
                mapping[str(df.columns[i])] = "proc_objects_one_hot"
            else:
                mapping[str(df.columns[i])] = "proc_objects_string"
        elif df.dtypes.iloc[i] == "int64":
            if (len(df[df.columns[i]].unique()) / len(df) <= THRESHOLD_RATIO_CATEGORIES) & \
                (len(df[df.columns[i]].unique()) <= THRESHOLD_ABSOLUTE_CATEGORIES):
                mapping[str(df.columns[i])] = "proc_objects_one_hot"
            else:
                mapping[str(df.columns[i])] = "proc_raw"
        elif df.dtypes.iloc[i] == "datetime64[ns]":
            #features.append( proc_datetime( df[df.columns[i]] ) )
            mapping[str(df.columns[i])] = "proc_datetime"
        else:
            print("WARNING: UNKNOWN COLUMN TYPE")
        
    return mapping


def auto_edgerizer(dfs):
    strong_edges = list()
    all_maps = dict()
    for name, df in dfs:
        curr_map = dict()
        for i in range(len(df.columns)):
            if (df.dtypes.iloc[i] == "object") | (df.dtypes.iloc[i] == "int64"):
                curr_map[df.columns[i]] = df[df.columns[i]].unique()
        all_maps[str(name)] = copy.copy(curr_map)
    
    # Now find overlaps
    keys = sorted(list(all_maps.keys()))
    for k in range(len(keys)):
        for l in range(len(keys)):
            if k < l:
                k_cols = sorted(list(all_maps[keys[k]].keys()))
                l_cols = sorted(list(all_maps[keys[l]].keys()))
                for kk in range(len(k_cols)):
                    for ll in range(len(l_cols)):
                        inter = set(all_maps[keys[k]][k_cols[kk]].tolist()).intersection( 
                            set(all_maps[keys[l]][l_cols[ll]].tolist() ))
                        if (len(set(all_maps[keys[k]][k_cols[kk]].tolist() )) == len(inter)) | \
                            (len(set(all_maps[keys[l]][l_cols[ll]].tolist() )) == len(inter)):
                            if len(set(all_maps[keys[l]][l_cols[ll]].tolist() )) == len(set(all_maps[keys[k]][k_cols[kk]].tolist() )) and \
                                len(set(all_maps[keys[l]][l_cols[ll]].tolist() )) == len(inter):
                                tmp_dict = dict()
                                tmp_dict["name"] = str(keys[k]) + "_" + str(keys[l]) 
                                tmp_dict["from"] = str(keys[k])
                                tmp_dict["to"] = str(keys[l])
                                tmp_dict["transform"] = ""
                                tmp_dict["from_col"] = str(k_cols[kk])
                                tmp_dict["to_col"] = str(l_cols[ll])
                                strong_edges.append(tmp_dict)
                                
    return strong_edges


def identify_key_column(data):
    for i in range(len(data.columns)):
        if (data.dtypes.iloc[i] == "object") or (data.dtypes.iloc[i] == "int64"):
            if len(data[data.columns[i]].unique()) == len(data):
                return str(data.columns[i])
    return ""


def create_auto_config_from_files(files, guidance=None, yaml_file="./src/default01.yml", projectname="basic"):
    """
    Create a yaml file as well as a python file to give a 
    customizable basis for any following pipeline
    """
    # 0. Read files and create dfs tuple-object
    dfs = []
    for entry in files:
        dfs.append((entry[(entry.rfind("/") + 1) : entry.rfind(".")], pd.read_parquet(entry)))

    # 1. Create Node-FeatureExtraction Mapping
    mappings = list()
    for i, tup in enumerate(dfs):
        name, df = tup
        obj = dict()
        obj["name"] = name
        obj["file"] = files[i]
        obj["key"] = identify_key_column(df)
        obj["label"] = identify_key_column(df)
        obj["features"] = "features"
        obj["transform"] = auto_enc(df)
        mappings.append(obj)

    # 2. Create Edge Mapping
    # strong_edges = auto_edgerizer(dfs)

    final_dict = dict()
    final_dict["project"] = projectname
    final_dict["data_dir"] = "./data"
    final_dict["backend"] = {"uri": final_dict["data_dir"] + "/" + projectname}
    final_dict["script"] = projectname + "_utils"
    final_dict["nodes"] = mappings
    #final_dict["edges"] = strong_edges

    # 3. Write yaml file
    f = open(yaml_file, "w")
    yaml.dump(final_dict, f, sort_keys=False)
    f.close()
    print("YAML file saved.")
    return


def run_feature_pipeline_from_yaml(file="./src/default01.yml"):
    config = load_yaml(file)
    module = __import__(config["script"])

    pp = getattr(module, "postproc")

    for entry in config["nodes"]:
        data = pd.read_parquet(entry["file"])
        feature_data = []
        for transform in entry["transform"]:
            func = entry["transform"][transform]
            f = getattr(module, func)
            feature_data.append(f(data[transform]))
        
        data[entry["features"]] = pp(feature_data).tolist()
        data.to_parquet(entry["file"])
    
    for entry in config["edges"]:
        for node in config["nodes"]:
            if node["name"] == entry["from"]:
                df = pd.read_parquet(node["file"])
                # Write parquet files of edges:
                df[[entry["from_col"], entry["to_col"]]].to_parquet(
                    config["data_dir"] + '/edges_' + entry["from"] + "_" + entry["to"] + ".parquet")
