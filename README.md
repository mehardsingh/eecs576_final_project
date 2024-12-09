# Modeling Time Dynamics in Sequential Recommendation

Traditional approaches for sequential recommendation primarily focus on the order of historical user interactions, neglecting the varying time intervals between purchases. In this paper, we introduce Temporal Linear Attention Bias (T-LAB), an enhancement to the BERT4Rec-style transformer model that incorporates time dynamics by applying linear attention biases based on timestamps. Our contributions are threefold. First, we investigate the performance of existing sequential recommendation models under non-uniform time gaps. Second, we evaluate the improvement brought about by T-LAB. Third, we explore the effectiveness of combining graphical inputs through a hybrid model that integrates Graph Neural Networks (GNNs) with T-LAB. Experimental results demonstrate that T-LAB significantly enhances performance and is more robust to inconsistent purchase histories than baseline methods. 

## Setup

### Download Data

Unzip and extract each data file for each month in https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store. Organize the files in a new directory called `data`:

```
EECS576_Final_Project/
├── data/
│   ├── original/
│   │   ├── 2019-Oct.csv
│   │   ├── 2019-Nov.csv
│   │   ├── ...
│   │   └── 2020-Mar.csv
│   └──
└──
```

### Environment Setup

Create a conda environment (Python 3.9) and install the packages in `requirements.txt`

## Training

To train the models, run:

```
python src/train_bert.py
python src/train_srgnn.py
python evaluate_bert_tlab.ipynb
python src/train_hybrid.py
```

The resulting models will save in the `results/<model_name>` directory. Note that they will overwrite the current contents of the results directory.

## Evaluation

To evaluate the models on test data, please run the following Jupyter notebooks:

```
evaluate_bert.ipynb
evaluate_srgnn.ipynb
evaluate_bert_tlab.ipynb
evaluate_hybrid.ipynb
```

The final performance can be found in `results/<model_name>/total_results.csv`

Evaluation metrics include Recall@X and MRR.