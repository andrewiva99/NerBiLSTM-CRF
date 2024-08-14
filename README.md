# NerBiLSTM-CRF

This work is devoted to the implementation of the **BiLSTM model with a CRF layer** from scratch and its training on the **NER task**. The model, as well as the basic functions 
required for data preprocessing, training and evaluation, are **implemented using only the PyTorch library**.

## Model architecture

The model consists of a two-layer BiLSTM and a CRF layer. We use [fastText](
https://fasttext.cc/docs/en/pretrained-vectors.html)([Simple English](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.zip)) as word embeddings.

![Model_graph1](https://github.com/user-attachments/assets/ad16b615-6b5b-462f-a4d4-aaf51d35c401)

## Data

We utilize [NER data](https://www.kaggle.com/datasets/rajnathpatel/ner-data) that includes 47,000 examples with 17 different labels. 
The training dataset accounts for 90% of the total examples available.

![class_dist](https://github.com/user-attachments/assets/1ebc22c3-4683-4d52-b0e3-ed5db6f70e5d)

## Training and evaluation

Since there is class imbalance, we use **weighted cross-entropy** as the loss function and **macro f1-score** as the quality metric.
Initially, we train the model **without the CRF layer** for 10 epochs and a batch size of 8. The model achieves a **macro f1-score of 76%** on the test.

![wo_crf](https://github.com/user-attachments/assets/083f43aa-a78c-4214-b348-0e77565fad98)

Then we **add a CRF layer** and train for 3 epochs with a batch size of 8. The model achieves a **macro f1-score of 81%** on the test.

![with_crf](https://github.com/user-attachments/assets/9970312c-3ede-46f9-8583-a85805227593)

## Project details

Implementation of the **BiLSTM with CRF layer**, data preprocessing, implementation of the main functions for training and evaluating the model 
are presented in jupyter notebook ```ner_bilstm_crf.ipynb```. The model implementation is separately presented in ```ner_bilstm_crf.py```. 
The trained weights of the ```NerBiLSTMCRF``` model are stored in ```ner_bilstm_crf_weights.pth```.
