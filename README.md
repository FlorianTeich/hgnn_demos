# Heterogeneous Graph Neural Network - Demoproject

## Abstract

In this project I want to leverage and demonstrate the advantages of HGNNs.
The project contains a routine to create synthetical data about people, their diagnoses and their drug-consumptions.
The target variable that is going to be predicted by the HGNN reflects whether a given person was diagnosed with a specific diagnosis and consumed a specific drug subsequently - a typical binary classification problem.

### Example:

Person 1:
* Consumed drugs: AB (3 years ago), HS (7 years ago), FO (5 years ago)
* Diagnoses: L32 (8 years ago), C80 (6 years ago)
* Target label: 1 (due to coexistence of diagnosis L32 and drug AB in their temporal order)

Person 1:
* Consumed drugs: YC, KI, PE, NN, XQ
* Diagnoses: R47, T20, N08
* Target label: 0

Since the source data is relational, a significant amount of time needs to be invested into Feature Engineering: As any Person may only have a certain target label but any amount of raw features (drugs consumed as well as diagnoses) the data would be required to be aggregated in order to generate features for each person.
Such aggregation of secondary entity features is risky as we possibly throw away critical information.
The advantage of HGNNs: we do not need any

### Data Structures

Person:
* Age
* Gender

Diagnosis:
* DiagnosisCode [A00 - Z99]
* DiagnosisAge

Drug:
* DrugCode [AA - ZZ]
* DrugAge

## Installation

```
git clone https://github.com/FlorianTeich/hgnn_demos
cd hgnn_demos
mkdir data
poetry shell
poetry install
cd src
python manage.py --help
```

Afterwards you need to install pyg_lib or torch_scatter, e.g. by running
```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cpu.html
```

## Usage
```
python manage.py --help

╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────╮
│ benchmark-transformer                  Train a Transformer model on the toy data                           │
│ explain                                Generate visual explanations to some of the test data               │
│ generate-toy-data                      Generate toy data                                                   │
│ migrate-data-to-kuzu                   Migrate Data to Kuzu                                                │
│ prepare-transformer-benchmark          Prepare data for Transformer model benchmark                        │
│ train-test-loop                        Train HGNN on the toy data                                          │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Results

The predefined HGNN should achieve between 95-99% accuracy on the test data.

## Explanations

After training the network we are able to use the IntegratedGradients in order to quantify how essential every node was to the overall prediction.

Coming soon...

## Benchmarks

Coming soon...
