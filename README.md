Implementation of the ESIM model for natural language inference with fastNLP

This repository contains an implementation with fastNLP of the sequential model presented in the paper "Enhanced LSTM for Natural Language Inference" by Chen et al. in 2016.



Download the SNLI corpus and the GloVe 840B 300d embeddings and put files in such structure:

DATA_DIR/
	   |-- data/
	   |    |-- checkpoints/
	   |    |    |-- best.pth.tar
	   |    |-- dataset/
	   |    |    |-- snil_1.0/
	   |    |         |-- README.txt
     |    |         |-- snli_1.0_dev.jsonl
     |    |         |-- snli_1.0_dev.txt
     |    |         |-- snli_1.0_test.jsonl
     |    |         |-- ...
     |    |-- embeddings/
     |    |    |-- glove.840B.300d.txt
	   |-- scripts/
	   |    |-- model.py
	   |    |-- preprocess_data.py
	   |    |-- train_model.py
	   |    |-- test_train.py
	   |    |-- utils.py
### Data Preparation

Download [FashionAI dataset](http://fashionai.alibaba.com/datasets/) and put files in such structure:

	DATA_DIR/
		   |-- wu_train/
		   |    |-- Annotations/
		   |    |    |-- annotations.csv
		   |    |-- Images/
		   |    |    |-- blouse
		   |    |    |-- ...
		   |-- r1_train/
		   |    |-- Annotations/
		   |    |    |-- annotations.csv
		   |    |-- Images/
		   |    |    |-- blouse
		   |    |    |-- ...
		   |-- ...
