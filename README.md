# chest-xray-disease-classification

### Introduction

The dataset consists of chest xray scan labeled with 14 different types of diseases. The idea is to train a deep learning model to identify the presence of any disease. This problem falls under multi label classification. Our model consists of a number of resnet layers followed by a dense layer of 14 units.

### How to handle class imbalance ?
This is a major challenge in this medical diagnosis related datasets. The number of negative samples by far outnumber the positive samples. 

Undersampling the majority class can reduce imbalance. Here, we have retained only 20% of those samples which do not have the presence of any disease.

A weighted loss function can be used to account for this huge class imbalance. We can modify the usual log loss ( binary crossentropy loss ) by taking a weighted average of loss due to the positive class and negative class. The classes are weighted by the frequency of the complementary class.

### What is our metric ?
Accuracy is not the right metric for medical diagnosis related problem. Due to huge class imbalance, a classifier can achieve high accuracy by predicting 0 for all samples without learning anything meaningful.

Hence, confusion matrix is of crucial importance. The threshold needs to be set carefully. As the threshold varies, there is a tradeoff between precision and recall. We need a single number to judge our model. Hence, the most appropriate metric is area under ROC curve(auc).

### The model
We have used a resnet v2 with depth 56 as the base model. A classifier with 14 units is placed on top of the base model. Sigmoid is used as the activation function for the output layer. The value output by sigmoid is interpreted as the probability of the occurence of that disease.

kaggle notebook : https://www.kaggle.com/abhinavjain02/chest-x-ray-disease-classification

This work has been inspired from :

https://www.kaggle.com/kmader/train-simple-xray-cnn

https://www.coursera.org/learn/ai-for-medical-diagnosis/home/welcome

https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/

The results and weights of this notebook are uploaded here :
https://www.kaggle.com/abhinavjain02/resnet-weights
