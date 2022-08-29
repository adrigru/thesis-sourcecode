# Evaluation of Machine Learning Approaches for Assessment of Reflective Level
## Abstract
Reflective writing is a process of revisiting an experience and evaluating it while focusing on emotions and what could have gone differently. It promotes critical thinking, strengthens the understanding of key concepts, and fosters personal growth. However, mastering it relies heavily on external feedback. Currently, students receive support for developing their sense of reflective writing from teachers. But this process is time-consuming, requires skilled personnel, and sometimes students do not enjoy sharing their thoughts with the teachers. 
For this reason, generating automated feedback on reflective writing would encourage students and unburden the educators. Previous studies show the potential of rule-based and machine learning approaches for various reflective writing analysis tasks, such as classifying reflective sentences. Yet they do not attempt to generate actionable feedback and use different data and annotation schemes, which makes it hard to compare the results. This thesis examines methods for classifying the depth of reflection using document-level, sentence-level, and feature engineering approaches on a single data set.
First, it evaluates commonly used techniques, such as Boosting and Logistic Regression, followed by Deep Transfer Learning methods. Second, it examines the performance of top-performing models on test data. The results show that document-level prediction using a pre-trained Transformer model outperforms the sentence-based classification by 0.043818 QWK and the feature-based approach by 0.0092 QWK for English. Similarly, the document-based prediction beats the sentence-level prediction by 0.17977 QWK for German. 

Repository contains source code for master thesis 

## Setup
Create virtual environment and install the dependencies
```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
# pip3 install -r requirements.txt
```
## Run training 
```bash
$ python3 train.py  --data_dir='../../data' --filename='../../data/feature_embeddings.tsv' --batch_size 64 --learning_rate 0.767842 -t_0 50 t_mult 2 --epochs 500
```