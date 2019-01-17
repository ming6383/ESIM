# ESIM

This repository contains an implementation with fastNLP of the sequential model presented in the paper "Enhanced LSTM for Natural Language Inference" by Chen et al. in 2016.

## Getting Started

### Dependency

- Python3.6
- Numpy
- Pytorch
- fastNLP

### Data Preparation

Download the [SNLI](https://nlp.stanford.edu/projects/snli/) corpus and
the [GloVe 840B 300d](https://nlp.stanford.edu/projects/glove/) embeddings and put files in such structure:

	     ESIM/
		   |-- data/
		   |    |-- checkpoints/
		   |    |    |-- best.pth.tar
		   |    |-- datasets/
		   |    |    |-- snil_1.0/
		   |    |         |-- README.txt
		   |    |         |-- snli_1.0_dev.jsonl
		   |    |         |-- snli_1.0_dev.txt
		   |    |         |-- snli_1.0_test.jsonl
		   |    |         |-- ...
		   |    |-- embeddings/
		   |    |    |-- glove.840B.300d.txt
		   |    |-- config.json
		   |-- scripts/
		   |    |-- model.py
		   |    |-- preprocess_data.py
		   |    |-- train_model.py
		   |    |-- test_model.py
		   |    |-- utils.py
		   |-- README.md

## Preprocess_data

Before the downloaded corpus and embeddings can be used in the ESIM model, they need to be preprocessed. This can be done with
the *preprocess_data.py* script in the *scripts/* folder of this repository. 

The script's usage is the following:
```
preprocess_data.py 
```

## Train the model

The *train_model.py* script can be used to train the ESIM model on some training data and validate it on some validation data.

The script's usage is the following:
```
train_model.py
```

## Test the model

The *test_model.py* script can be used to test the model on some test data.

The script's usage is the following:
```
test_model.py
```
## Results

A pretrained model is made available in the *data/checkpoints* folder of this repository. The model was trained with the
parameters defined in the default configuration files provided in *data*.

The pretrained model achieves the following performance on the SNLI dataset:

| Split | Accuracy (%) |
|-------|--------------|
| Train |     93.2     |
| Dev   |     88.4     |
| Test  |     88.0     |

The results are in line with those presented in the paper by Chen et al.
