# TopK+L2D
This code is for the paper ``Efficient Demonstration Selection by Label-Alignment Divergence Reranking for In-Context Learning.

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
