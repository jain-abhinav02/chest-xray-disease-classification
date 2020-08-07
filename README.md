# Chest X-Ray Disease Classification

### Quick Links
Kaggle dataset : https://www.kaggle.com/nih-chest-xrays/data

Kaggle notebook : https://www.kaggle.com/abhinavjain02/chest-x-ray-disease-classification

The results and weights of this notebook are uploaded here : https://www.kaggle.com/abhinavjain02/resnet-weights

---

### Team members 

1. [Abhinav Jain](https://github.com/jain-abhinav02)

1. [Aayushman Mishra](https://github.com/mishraaayushman3)

---

### Introduction

The dataset consists of chest xray scans labeled with 14 different types of diseases. The idea is to train a deep learning model to identify the presence of a disease. Since a patient may have symptoms of multiple diseases, this problem falls under multi label classification. The dataset consists of 112,120 x-ray images of 30,805 unique patients.

---

### How to handle class imbalance ?
This is a major challenge in this medical diagnosis related datasets. The number of negative samples by far outnumber the positive samples. 

Undersampling the majority class can reduce imbalance. Here, we have retained only 20% of those samples which do not have the presence of any disease.

A weighted loss function can be used to account for this huge class imbalance. We can modify the usual log loss ( binary crossentropy loss ) by taking a weighted average of loss due to the positive class and negative class. The classes are weighted by the frequency of the other class.

---

### What is the right metric ?
Accuracy is not the right metric for medical diagnosis related problem. Due to huge class imbalance, a classifier can achieve high accuracy by predicting 0 for all samples without learning anything meaningful.

Hence, confusion matrix is of crucial importance. The threshold needs to be set carefully. As the threshold varies, there is a tradeoff between precision and recall. It is convenient to have a single number to judge our model. Hence, the most appropriate metric would be area under ROC curve(auc).

---

### The model
We have used a resnet v2 with depth 56 as the base model. A classifier with 14 units is placed on top of the base model. Sigmoid is used as the activation function for the output layer. The value output by sigmoid is interpreted as the probability of the occurence of that disease. We would encourage you to experiment with other pre-trained networks as base models. One might also use weights of models pre-trained on image net dataset.

The dataset is split randomly such that 80% is used for training, 5% for validation and 15% for testing.The neural network was trained for a total of 30 epochs. However, analysis of results on the validation set shows that model generalizes better at the end of 20th epoch. Hence, the weights at the end of 20th epoch is used for testing.

---

### Loss

<img src="/visualisation/loss.png" width="600" height="400">

### Training set statistics

<img src="/visualisation/training_set_roc_curve_epoch_20.png" width="500" height="500">

### Test set statistics

| Metric        | Value          |
|:--------------------:|:------------------:|
| Min AUC | 0.5843 |
| Max AUC | 0.8115 |
| Mean AUC | 0.7055 |
| Median AUC | 0.7021 |  

* ROC Curve

<img src="/visualisation/test_set_roc_curve.png" width="500" height="500">

---

### Grad CAM
Gradient weighted class activation mapping
It uses the gradients of any target concept flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept. Grad CAM is a great tool to visualize what the CNN model is learning.

<img src="/visualisation/xray_samples.png" width="600" height="600">

---

### Key Learning

* Techniques to handle class imbalance
* Custom loss function in keras
* ImageData Generator 
* Receiver operating characteristic (ROC)
* Gradient weighted class activation mapping (Grad CAM)

---

### References
This work has been inspired from :

* https://www.kaggle.com/kmader/train-simple-xray-cnn

* https://www.coursera.org/learn/ai-for-medical-diagnosis/home/welcome

* https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/