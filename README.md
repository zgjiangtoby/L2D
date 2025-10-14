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






## Compute-accuracy trade-off:

We test the average inference GPU-hours of each method on seven datasets, **L2D 0.96 < MDL 1.06 < ConE 1.14**. The accuracy improvements of L2D are larger on ambiguous tasks (e.g., **+5.39% SST-5, +4.87% MNLI, +4.4% Subj**).

| Method | AgNews | CR | SST-2 | SST-5 | Subj | MNLI | QNLI | Infer Time |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| MDL | 77.83(↑0.37) | 94.41(↑0.27) | 96.05(↑0.44) | 50.05(↑4.25) | 92.35(↑2.80) | 79.90(↑3.58) | 82.76(↑2.69) | 1.06(+0.10) |
| ConE | 80.95(↓2.75) | 93.88(↑0.80) | 95.61(↑0.88) | 48.91(**↑5.39**) | 90.75(↑4.40) | 78.61(↑4.87) | 84.26(↑1.19) | 1.14(**+0.18**) |
| L2D | 78.20(↑0) | 94.68(↑0) | 96.49(↑0) | 54.30(↑0) | 95.15(↑0) | 83.48(↑0) | 85.45(↑0) | 0.96(+0.00) |

Note: "↑" and "+" denote the accuracy improvements and the extra GPU-Hours respectively, compared with L2D. **Bold** indicate the largest gain.

### Inference time
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


### Pre-trained Language Models size

| Model                     | #Params (M) |
|----------------------------|--------------------|
| BERT-base-uncased          | 86                 |
| RoBERTa-base               | 86                 |
| DeBERTa-v3-base            | 86                 |
- Base models:12 layers,768 hidden size,12 heads
- M denotes for millions parameters
### Reference
He, P., Gao, J., | Chen, W. DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. In The Eleventh International Conference on Learning Representations. 2023



