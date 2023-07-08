import typer

import datetime
import multiprocessing as mp
import os.path as osp
import string
import urllib
from multiprocessing import cpu_count
from os import path
import utils
import click
import kuzu
import numpy as np
import pandas as pd
import sqlalchemy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sqlalchemy import create_engine
from torch import Tensor
from torch.nn import Linear
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch, Dataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import (MLP, BatchNorm, GraphConv, MultiAggregation,
                                SAGEConv, to_hetero)
from tqdm import tqdm
import artificial_data
import pathlib

app = typer.Typer()

@app.command()
def generate_toy_data():
    """Generate toy data"""
    NUM_SAMPLES=100000
    sqlite_file_path = str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/backend.db"
    print(sqlite_file_path)
    backend = create_engine("sqlite:///" + sqlite_file_path)
    artificial_data.generate_all_tables(person_count=NUM_SAMPLES, conn=backend)
    persons = pd.read_sql_table("persons", con=backend)
    diagnosis_data = pd.read_sql_table("diagnoses", con=backend)
    drug_data = pd.read_sql_table("drugs", con=backend)

    train, validate, test = np.split(persons.sample(frac=1), [int(.6*len(persons)), int(.8*len(persons))])
    persons.loc[train.index, "mode"] = "train"
    persons.loc[validate.index, "mode"] = "validate"
    persons.loc[test.index, "mode"] = "test"

    # Create Short Codes
    diagnosis_data["diagnosis_short"] = diagnosis_data["DiagnosisCode"].str[:1]
    drug_data["drug_short"] = drug_data["DrugCode"].str[:1]

    scaler2 = StandardScaler()
    diagnosis_data = pd.read_sql_table("diagnoses", con=backend)

    # Get gender and age
    le = preprocessing.LabelEncoder()
    le.fit(persons[persons["mode"] != "test"].Gender)
    persons["gender_encoded"] = le.fit_transform(persons.Gender)

    persons["now"] = pd.to_datetime(datetime.datetime.now())
    persons["age"] = (persons.now - persons.Birthday).dt.days
    persons["age"] = scaler2.fit_transform(np.expand_dims(pd.to_numeric(persons["age"], downcast="float"), 1))

    # Get diagnosis codes
    scaler = MaxAbsScaler()
    diagnosis_data = diagnosis_data.merge(persons[["PID", "Birthday", "mode"]], on="PID", how="left")
    diagnosis_data["diagnosis_age"] = (diagnosis_data["DiagnosisDate"] - diagnosis_data.Birthday).dt.days
    scaler.fit(diagnosis_data[diagnosis_data["mode"] != "test"][["diagnosis_age"]])
    diagnosis_data["diagnosis_age"] = scaler.transform(diagnosis_data[["diagnosis_age"]])
    diagnosis_data["diagnosis_short"] = diagnosis_data["DiagnosisCode"].str[:1].apply(str.lower)
    diagnosis_agg = diagnosis_data.groupby("PID")['diagnosis_short'].apply(" ".join)

    merged_diagnosis = persons.merge(diagnosis_agg, how="left", on="PID")

    # Get drug codes
    scaler2 = MaxAbsScaler()
    drug_data = drug_data.merge(persons[["PID", "Birthday", "mode"]], on="PID", how="left")
    drug_data["drug_age"] = (drug_data["DrugDate"] - drug_data.Birthday).dt.days
    scaler2.fit(drug_data[drug_data["mode"] != "test"][["drug_age"]])
    drug_data["drug_age"] = scaler2.transform(drug_data[["drug_age"]])
    drug_data["drug_short"] = drug_data["DrugCode"].str[:1].apply(str.lower)
    drug_agg = drug_data.groupby("PID")['drug_short'].apply(" ".join)

    merged_drug = persons.merge(drug_agg, how="left", on="PID")

    def analyzer_custom(doc):
        return doc.split()

    vectorizer_diagnosis = CountVectorizer(vocabulary=list(string.ascii_lowercase),
                                analyzer=analyzer_custom)
    vectorizer_drug = CountVectorizer(vocabulary=list(string.ascii_lowercase),
                                analyzer=analyzer_custom)

    data_diagnosis = vectorizer_diagnosis.transform(merged_diagnosis["diagnosis_short"].values.astype('U')).todense()
    data_drug = vectorizer_drug.transform(merged_drug["drug_short"].values.astype('U')).todense()
    merged_diagnosis.loc[:, "diagnosis_vec"] = data_diagnosis.tolist()
    merged_drug.loc[:, "drug_vec"] = data_drug.tolist()

    X = np.hstack([merged_diagnosis[["gender_encoded", "age"]].values, 
                np.vstack(merged_diagnosis["diagnosis_vec"].values),
                np.vstack(merged_drug["drug_vec"].values)])

    # Create Target Labels
    diagnosis_data["label_i"] = ((diagnosis_data["diagnosis_short"] <= "k") & (diagnosis_data["diagnosis_age"] < 0))
    drug_data["label_a"] = ((drug_data["drug_short"] <= "k") & (drug_data["drug_age"] > 0))
    persons = persons.merge(diagnosis_data[["PID", "label_i"]].groupby("PID").max(), on="PID", how="left")
    persons = persons.merge(drug_data[["PID", "label_a"]].groupby("PID").max(), on="PID", how="left")
    persons["label"] = (persons["label_i"] == True) & (persons["label_a"] == True)
    print(persons.label.value_counts())

    data = np.hstack([utils.encode_strings(diagnosis_data["diagnosis_short"]), diagnosis_data[["diagnosis_age"]].values])
    feats = pd.DataFrame(((x,) for x in data), columns=["features"])
    diagnosis_data = pd.concat([diagnosis_data, feats], axis=1)

    data = np.hstack([utils.encode_strings(drug_data["drug_short"]), drug_data[["drug_age"]].values])
    feats = pd.DataFrame(((x,) for x in data), columns=["features"])
    drug_data = pd.concat([drug_data, feats], axis=1)

    data = persons[["age", "gender_encoded"]].values
    feats = pd.DataFrame(((x,) for x in data), columns=["features"])
    persons = pd.concat([persons, feats], axis=1)

    persons.to_parquet(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/persons.parquet")
    diagnosis_data.to_parquet(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/diagnoses.parquet")
    drug_data.to_parquet(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/drugs.parquet")


@app.command()
def migrate_data_to_kuzu():
    """Migrate Data to Kuzu"""
    db = kuzu.Database(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/demo")
    conn = kuzu.Connection(db, num_threads=cpu_count())
    persons = pd.read_parquet(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/persons.parquet")
    diagnosis_data = pd.read_parquet(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/diagnoses.parquet")
    drug_data = pd.read_parquet(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/drugs.parquet")

    utils.kuzu_node_table_from_arrays(conn, tablename="drug", feats=np.stack(drug_data["features"].values),
                        labels=drug_data["label_a"].values)

    utils.kuzu_node_table_from_arrays(conn, tablename="diagnosis", feats=np.stack(diagnosis_data["features"].values),
                                labels=diagnosis_data["label_i"].values)

    utils.kuzu_node_table_from_arrays(conn, tablename="person", feats=np.stack(persons["features"].values),
                                labels=persons["label"].values)

    utils.kuzu_edges_from_tensor(conn, np.flip(persons.merge(diagnosis_data, on="PID", how="inner")[["index_x", "index_y"]].values.T), "assigned_to", "diagnosis", "person")
    utils.kuzu_edges_from_tensor(conn, np.flip(persons.merge(drug_data, on="PID", how="inner")[["index_x", "index_y"]].values.T), "consumed_by", "drug", "person")


@app.command()
def train_test_loop():
    """Train HGNN on the toy data"""
    db = kuzu.Database(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/demo")
    persons = pd.read_parquet(str(pathlib.Path(__file__).parent.parent.resolve()) + "/data/persons.parquet")
    train_inds = torch.tensor(persons[persons["mode"] == "train"].index.tolist(), dtype=torch.long)
    test_inds = torch.tensor(persons[persons["mode"] == "test"].index.tolist(), dtype=torch.long)
    train_loader, test_loader = utils.get_loaders(db, 32, train_inds, test_inds)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    its = iter(train_loader)
    batch = next(its)

    model, optimizer = utils.get_model_v1(device, batch)

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
                loss = F.cross_entropy(out, btch["person"].y.long().view(-1))
                loss.backward()
                optimizer.step()
                preds.extend(torch.argmax(torch.softmax(out, dim=-1), dim=-1).detach().tolist())
                ytrue.extend(btch["person"].y.tolist())
                all_loss += loss.item() / len(btch["person"].y.tolist())
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
                loss = F.cross_entropy(out, btch["person"].y.long().view(-1))
                preds.extend(torch.argmax(torch.softmax(out, dim=-1), dim=-1).detach().tolist())
                ytrue.extend(btch["person"].y.tolist())
                all_loss += loss.item() / len(btch["person"].y.tolist())
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

    for i in tqdm(range(1)):
        train(i)
        test(i)
    
    torch.save(model.state_dict(), str(pathlib.Path(__file__).parent.parent.resolve()) + "/models/checkpoint.pt")


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
    import torch
    from tqdm.auto import tqdm
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification
    from transformers import AutoTokenizer, DistilBertConfig
    from torch.utils.data import Dataset
    from torch.nn import functional as F
    from tqdm import tqdm
    import numpy as np
    import itertools
    import matplotlib.pyplot as plt
    import pandas as pd
    from datasets import load_metric
    from torch.nn import functional as F
    from datasets import Dataset, load_dataset, load_from_disk

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
