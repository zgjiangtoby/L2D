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

| Dataset | L2D | ConE | MDL |
|:---------|----:|----:|----:|
| ag_news | 2041 | 2060 | 2477 |
| cr | 67 | 123 | 83 |
| mnli | 6924 | 8811 | 7101 |
| qnli | 2070 | 2018 | 2299 |
| sst2 | 285 | 374 | 408 |
| sst5 | 371 | 477 | 527 |
| subj | 380 | 488 | 549 |
| **AVG (seconds)** | **1734** | **2050.14** | **1920.57** |
| **GPU-Hours** | **0.96** | **1.14** | **1.06** |

Note: GPU-Hours are calculated based on two NVIDIA RTX-4090 GPUs, i.e., 1734/360 * 2 = 0.96.

## Compute-accuracy trade-off:

|Method| AgNews | CR | SST-2 | SST-5 | Subj | MNLI | QNLI | Infer Time |
|:---------|----:|----:|----:|----:|----:|----:|----:| ----:|
|MDL| 77.83 |94.41 |96.05 | 50.05 | 92.35 | 79.90 | 82.76 | 1.06 (+0.1) |
|ConE| 80.95 | 93.88 | 95.61 | 48.91 | 90.75 | 78.61 |84.26 | 1.14 (**+0.18**) |
|L2D| 78.20 ($\uparrow$ 0) | 94.68 ($\uparrow$ 0) | 96.49 ($\uparrow$ 0) | 54.30 ($\uparrow$ 0) | 95.15 ($\uparrow$ 0) | 83.48 ($\uparrow$ 0) | 85.45 ($\uparrow$ 0) | **0.96**(+0) |

## Pre-trained Language Models size

| Model                     | Backbone #Params(M) |
|----------------------------|--------------------|
| BERT-base-uncased          | 86                 |
| RoBERTa-base               | 86                 |
| DeBERTa-v3-base            | 86                 |
- Base models:12 layers,768 hidden size,12 heads
- M denotes for millions parameters
### Reference
He, P., Gao, J., | Chen, W. DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. In The Eleventh International Conference on Learning Representations. 2023



