# TabSim

### A Siamese Neural Network for Accurate Estimation of Table Similarity

This repository contains a table corpus made from tables extracted from scientific articles published in PMC, and additionally contains the code and the models for measuring the similarity between a pair of tables. You can also reproduce our results on tables from <a href = "https://github.com/bfetahu/wiki_tables"> Wikipedia</a> and <a href = "http://boston.lti.cs.cmu.edu/eager/table-arxiv/"> arXiv </a> articles.
<br>
## Train a model
To train a model, you need to use the TabSimTrain.py and provide the number of epochs, the learning rate value and the location of the trained embeddings (download it from <a href = "http://bio.nlplab.org/"> here</a>), the table data (in **tables.pickle** file) and the directory of the trained model:
```
TabSimTrain.py [-h] -e N_EPOCH -l LR -v EMBEDDING_LOC -i INPUT_TABLES -o MODEL_DIR
```


## Table Similarity Score

The following command measures the similarity of the table query with the tables in the repository. Before using this command you need to train a model.

```
TabSimEval.py [-h] -m MODEL -i INPUT_TABLES -o OUTPUT_TAGS
```


## Citation
Please cite the following work
```
@article{habibi2020tabsim,
  title={TabSim: A Siamese Neural Network for Accurate Estimation of Table Similarity},
  author={Habibi, Maryam and Starlinger, Johannes and Leser, Ulf},
  journal={arXiv preprint arXiv:2008.10856},
  year={2020}
}
```

