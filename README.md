# Heterogeneous Graph Neural Network - Demoproject

## Abstract

In this project I want to leverage and demonstrate the advantages of HGNNs.
The project contains a routine to create synthetic data or download toy data.

## Installation

```
git clone https://github.com/FlorianTeich/hgnn_demos
cd hgnn_demos
mkdir data
poetry shell
poetry install
poetry run python src/scripts/manage.py --help
```

Afterwards you need to install pyg_lib or torch_scatter, e.g. by running
```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cpu.html
```

## Usage
```
python manage.py --help

╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ convert-relational-db-to-graph-db                     Convert relational database to graph database               │
│ generate-toy-data-in-relational-db                    Generate toy data                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

## Results

The predefined HGNN should achieve between 95-99% accuracy on the test data.

## Explanations

After training the network we are able to use the IntegratedGradients in order to quantify how essential every node was to the overall prediction.

Coming soon...

## Benchmarks

Coming soon...
