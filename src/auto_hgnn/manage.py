import datetime
import multiprocessing as mp
import pathlib
from multiprocessing import cpu_count
from os import path

import kuzu
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import typer
from rich import print
from sklearn.metrics import accuracy_score
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import to_hetero
from tqdm import tqdm

from . import artificial_data
from . import utils
from .utils import load_yaml

app = typer.Typer()

@app.command()
def generate_toy_data(num_samples: int=50000):
    """Generate toy data"""
    artificial_data.generate_toy_datasamples(num_samples)


@app.command()
def migrate_data_to_kuzu(yamlfile="default.yml"):
    """Migrate Data to Kuzu"""
    # TODO: Report progress
    import os
    import sys
    config = load_yaml(yamlfile)
    db = kuzu.Database(config["backend"]["uri"])
    conn = kuzu.Connection(db, num_threads=cpu_count())

    for nodetype in config["nodes"]:
        nodedata = pd.read_parquet(nodetype["file"])
        utils.kuzu_node_table_from_arrays(conn,
                                          tablename=nodetype["name"],
                                          feats=np.stack(nodedata[nodetype["features"]].values),
                                          labels=nodedata[nodetype["label"]].values,
                                          path=config["data_dir"])
    
    sys.path.append(os.path.abspath(yamlfile[:yamlfile.rfind("/") + 1] + config["script"][:(config["script"].rfind("/") + 1)]))
    module = __import__(config["script"][(config["script"].rfind("/") + 1):])
    
    for edgetype in config["edges"]:
        func = getattr(module, edgetype["transform"])
        edges = func()
        utils.kuzu_edges_from_tensor(conn, edges, edgetype["name"], edgetype["from"], edgetype["to"],
                                     path=config["data_dir"])
    

@app.command()
def train_test_loop(yamlfile="default.yml"):
    """Train HGNN on the toy data"""
    config = load_yaml(yamlfile)
    db = kuzu.Database(config["backend"]["uri"])

    target_entity = config["task"]["target_entity"]

    train_inds = torch.tensor(pd.read_parquet(config["task"]["train_inds"])["ids"].values, dtype=torch.long)
    test_inds = torch.tensor(pd.read_parquet(config["task"]["test_inds"])["ids"].values, dtype=torch.long)
    
    train_loader, test_loader = utils.get_loaders(db, config["task"]["batch_size"], train_inds, test_inds, config["task"]["target_entity"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    its = iter(train_loader)
    batch = next(its)
    batch = batch.to(device)
    metadata = batch.metadata()

    module = __import__(config["script"])
    mdl = getattr(module, config["task"]["model"])
    margs = config["task"]["model_args"]
    hmargs = config["task"]["heteromodel_args"]
    model = mdl(**margs)
    model = to_hetero(model, metadata, **hmargs).to(device)

    opt = getattr(optim, config["task"]["optimizer"])
    oargs = config["task"]["optimizer_args"]
    optimizer = opt(model.parameters(), **oargs)

    out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)

    writer = SummaryWriter(str(pathlib.Path(__file__).parent.parent.resolve()) + "/runs/" + str(datetime.datetime.now()))

    def train(epoch=0):
        model.train()
        preds = []
        ytrue = []
        all_loss = 0.0
        for btch in train_loader:
            try:
                btch = btch.to(device, non_blocking=True)
                optimizer.zero_grad()
                out = model(btch.x_dict, btch.edge_index_dict, btch.batch_dict)
                loss = F.cross_entropy(out, btch[target_entity].y.long().view(-1))
                loss.backward()
                optimizer.step()
                preds.extend(torch.argmax(torch.softmax(out, dim=-1), dim=-1).detach().tolist())
                ytrue.extend(btch[target_entity].y.tolist())
                all_loss += loss.item() / len(btch[target_entity].y.tolist())
            except:
                pass
        acc = accuracy_score(ytrue, preds)
        #print("Train Loss:", all_loss)
        #print("Train Acc:", acc)
        writer.add_scalar("Loss/train", all_loss, epoch)
        writer.add_scalar("Acc/train", acc, epoch)
        all_loss = 0.0
        preds = []
        ytrue = []

    @torch.no_grad()
    def test(epoch=0):
        model.eval()
        preds = []
        ytrue = []
        all_loss = 0.0
        for btch in test_loader:
            try:
                btch = btch.to(device, non_blocking=True)
                out = model(btch.x_dict, btch.edge_index_dict, btch.batch_dict)
                loss = F.cross_entropy(out, btch[target_entity].y.long().view(-1))
                preds.extend(torch.argmax(torch.softmax(out, dim=-1), dim=-1).detach().tolist())
                ytrue.extend(btch[target_entity].y.tolist())
                all_loss += loss.item() / len(btch[target_entity].y.tolist())
            except:
                pass
        acc = accuracy_score(ytrue, preds)
        #print("Test Loss:", all_loss)
        #print("Test Acc:", acc)
        writer.add_scalar("Loss/test", all_loss, epoch)
        writer.add_scalar("Acc/test", acc, epoch)
        all_loss = 0.0
        preds = []
        ytrue = []

    for i in tqdm(range(5)):
        train(i)
        test(i)
    
    torch.save(model.state_dict(), config["task"]["model_checkpoints"])


@app.command()
def explain():
    """Generate visual explanations to some of the test data"""
    pass

@app.command()
def prepare_transformer_benchmark():
    """Prepare data for Transformer model benchmark"""
    persons = pd.read_parquet(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/persons.parquet")
    diagnosis_data = pd.read_parquet(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/diagnoses.parquet")
    drug_data = pd.read_parquet(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/drugs.parquet")

    diagnosis_data["diagnosis_age"] = diagnosis_data["diagnosis_age"].astype(str)
    diagnosis_agg = diagnosis_data.loc[:, ["PID", "DiagnosisCode", "diagnosis_age"]].groupby("PID").agg(lambda x: ''.join(x.str[0]))\

    drug_data["drug_age"] = drug_data["drug_age"].astype(str)
    drug_agg = drug_data.loc[:, ["PID", "DrugCode", "drug_age"]].groupby("PID").agg(lambda x: ''.join(x.str[0]))

    final_df = persons.loc[:, ["PID", "label", "mode"]].merge(diagnosis_agg, on="PID").merge(drug_agg, on="PID")

    diagnosis_hist = [" ".join(["$" + i + j for i, j in zip(final_df["DiagnosisCode"].values[k], final_df["diagnosis_age"].values[k])]) for k in range(len(final_df))]
    drug_hist = [" ".join(["#" + i + j for i, j in zip(final_df["DrugCode"].values[k], final_df["drug_age"].values[k])]) for k in range(len(final_df))]

    final_df["history"] = [i + " " + j for i, j in zip(diagnosis_hist, drug_hist)]
    final_df["label"] = final_df["label"].astype(int)
    final_df.to_parquet(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/final.parquet")

@app.command()
def benchmark_transformer():
    """Train a Transformer model on the toy data"""
    import itertools

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from datasets import Dataset, load_dataset, load_from_disk, load_metric
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
    from tqdm.auto import tqdm
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, DistilBertConfig)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    final_df = pd.read_parquet(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/final.parquet")

    MAX_LEN = 192
    modelname = "distilbert-base-uncased"
    tokenizername = "google/byt5-small"
    epochs = 10
    batch_size = 8

    settings = DistilBertConfig(n_heads=4, n_layers=4, num_labels=2)
    cmodel = AutoModelForSequenceClassification.from_pretrained(modelname, config=settings)
    tokenizer = AutoTokenizer.from_pretrained(tokenizername)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})

    cmodel.resize_token_embeddings(len(tokenizer))

    cmodel = cmodel.to(device)

    optimizer = torch.optim.AdamW(cmodel.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=epochs)
    loss_func = torch.nn.CrossEntropyLoss()

    def tokenize_function(examples):
        return tokenizer(examples["history"], padding="max_length", truncation=True, max_length=MAX_LEN)

    tokenized_datasets_train =  Dataset.from_pandas(final_df[final_df["mode"]=="train"].loc[:, ["label", "history"]], preserve_index=False).map(tokenize_function, batched=True)
    tokenized_datasets_test = Dataset.from_pandas(final_df[final_df["mode"]=="test"].loc[:, ["label", "history"]], preserve_index=False).map(tokenize_function, batched=True)

    tokenized_datasets_train = tokenized_datasets_train.rename_column("label", "labels")
    tokenized_datasets_train = tokenized_datasets_train.remove_columns("history")
    tokenized_datasets_test = tokenized_datasets_test.rename_column("label", "labels")
    tokenized_datasets_test = tokenized_datasets_test.remove_columns("history")

    tokenized_datasets_train.set_format("torch")
    tokenized_datasets_test.set_format("torch")

    small_train_dataset = tokenized_datasets_train.shuffle(seed=42)
    small_test_dataset = tokenized_datasets_test.shuffle(seed=42)

    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(small_test_dataset, batch_size=batch_size)

    #################################### DATASETS ##########################################################

    def train_all():
        cmodel.train()
        all_losses = []
        for i, batch in tqdm(enumerate(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            prediction = cmodel(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            loss = loss_func(prediction["logits"], batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            all_losses += [loss]
        return {"train_losses": all_losses}

    def test_all():
        all_preds = []
        all_targets = []
        all_losses = []
        cmodel.eval()
        for batch in tqdm(test_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                prediction = cmodel(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                loss = loss_func(prediction["logits"], batch["labels"])
            all_preds += [prediction]
            all_targets += [batch["labels"]]
            all_losses += [loss]
        
        return {"test_preds": all_preds, "test_targets": all_targets, "test_losses": all_losses}


    train_res = []
    test_res = []
    for e in range(epochs):
        train_res.append(train_all())
        test_res.append(test_all())
        acc = 0
        for i in range(len(test_res[-1]["test_preds"])):
            acc += (test_res[-1]["test_preds"][i]["logits"].argmax(1) == test_res[-1]["test_targets"][i]).sum()
        print(acc)

def benchmark_xgboost():
    """Train an XGBoost model on the toy data"""
    pass

if __name__ == '__main__':
    app()
