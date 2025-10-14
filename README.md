# TopK+L2D
This code is for the paper ``Learn to Select: Exploring Label Distribution Divergence for Demonstration Selection in In-Context Classification``.

# Installation

## Installation for local development:

```shell
git clone https://github.com/zgjiangtoby/L2D.git
cd L2D
pip install -e .
```

## Examples usage

1. prepares the datasets dirs as follows:
```shell
├── all_datasets
│   ├── cr_data
│   │   ├── train.csv
│   │   ├── test.csv
│   ├── SST2_data
│   │   ├── train.csv
│   │   ├── test.csv

....
```
2. change the pathes of LLMs, SLMs and retriever dirs in ``run_l2d.sh``.
3. run ``./run_l2d.sh``


## Inference time

| Dataset | L2D | Cone | MDL |
|:---------|----:|----:|----:|
| ag_news | 2041 | 2060 | 2477 |
| cr | 67 | 123 | 83 |
| mnli | 6924 | 8811 | 7101 |
| qnli | 2070 | 2018 | 2299 |
| sst2 | 285 | 374 | 408 |
| sst5 | 371 | 477 | 527 |
| subj | 380 | 488 | 549 |
| **AVG (seconds)** | **1734** | **2050.14** | **1920.57** |
| **2 GPU** | **0.96** | **1.14** | **1.06** |
