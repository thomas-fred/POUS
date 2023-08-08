Exploring poweroutage.us (POUS) electricity outage data

## Installation

Using [micromamba](https://mamba.readthedocs.io/en/latest/installation.html) for environment creation:
```
micromamba create -f ./env.yml -y
```

## Usage

First activate your environment:
```
micromamba activate POUS
```

Run the preprocessing script to read the raw CSV files located in `./data/raw`,
calculate the OutageFraction and save the results as parquet. See
`./data/processed/` for the output parquet files.
```
python preprocess.py
```

To start a notebook server and expore plotting outages:
```
jupyter notebook plot_outage.ipynb
```
