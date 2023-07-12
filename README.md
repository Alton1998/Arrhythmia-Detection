# Arrhythmia Detection

## Introduction

We will be using the [MIT data](https://archive.physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm) to train a model to detect Arrhythmias.

## Summary of the data after analyzing.

- **No. of records**: 48
- **Frequency** : 360 samples per second
- **Distribution**: 25 male subjects between the ages of 32 and 89, 22 female subjects aged from 23 to 89 years. 60% of the total subjects were inpatient.

## Aim
We will focus on classification of 5 classes, namely:

1. Normal (N)
2. Paced Beat (/)
3. Right Bundle Branch Block Beat (R).
4. Left Bundle Branch Block (L).
5. Premature Ventricular Beat (V)

## Type of problem

1. Classification problem
2. Supervised Learning problem.

## Tools And Frameworks
- [WFDB](https://wfdb.readthedocs.io/en/latest/)
- [Librosa](https://librosa.org/doc/latest/index.html)
- [Pytorch](https://pytorch.org/)
- [Pandas](https://pandas.pydata.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [py-ecg-detector](https://pypi.org/project/py-ecg-detectors/)
- [imbalanced-learn](https://imbalanced-learn.org/stable/)

## Steps to Implement in Code
1. Break each record into fragments by detecting the peaks and taking a window before and after the peak
2. Tranform the peak into feature vectors such that the number of dimensions is less than the original
3. Fit the data to a model
4. Report metrics

## Approaches

### Dimensionality Reduction Techniques
There are many dimensionality reduction methods, but some of the most common include:

* **Principal component analysis (PCA)** is a linear dimensionality reduction method that projects data points onto a lower-dimensional subspace in such a way that the variance of the data in the new space is maximized. This means that PCA finds the directions in the data that have the most variation, and projects the data points onto those directions.
* **Factor analysis (FA)** is another linear dimensionality reduction method that is similar to PCA. However, FA is designed to find latent factors that explain the variation in the data. These factors are not necessarily orthogonal, as they can be correlated with each other.
* **Linear discriminant analysis (LDA)** is a linear dimensionality reduction method that is specifically designed for classification tasks. LDA projects data points onto a lower-dimensional subspace in such a way that the classes are as well-separated as possible.
* **Non-negative matrix factorization (NMF)** is a nonlinear dimensionality reduction method that is often used for text analysis. NMF decomposes a matrix into two matrices, where the rows of the first matrix represent the original data points and the columns of the second matrix represent the latent factors.
* **Sparse coding** is a nonlinear dimensionality reduction method that is often used for image analysis. Sparse coding decomposes an image into a set of basis images, where each basis image is weighted by a coefficient. The coefficients are typically sparse, meaning that most of them are zero.

These are just a few of the many dimensionality reduction methods that are available. The best method to use for a particular task will depend on the nature of the data and the goals of the analysis.

Here are some of the benefits of using dimensionality reduction methods:

* **Reduced computational complexity:** When you reduce the dimensionality of your data, you also reduce the computational complexity of tasks such as clustering, classification, and regression. This can be a major advantage when you are working with large datasets.
* **Improved visualization:** When you reduce the dimensionality of your data, you can often visualize it more easily. This can be helpful for understanding the relationships between different variables and for identifying patterns in the data.
* **Improved performance:** In some cases, dimensionality reduction can actually improve the performance of machine learning models. This is because dimensionality reduction can help to remove noise from the data and to make the data more regular.

However, there are also some potential drawbacks to using dimensionality reduction methods:

* **Loss of information:** When you reduce the dimensionality of your data, you lose some of the information in the original data. This can be a problem if the information that is lost is important for the task that you are trying to perform.
* **Data distortion:** In some cases, dimensionality reduction can distort the data. This can make it more difficult to interpret the results of your analysis.
* **Overfitting:** If you reduce the dimensionality of your data too much, you can end up overfitting your model to the training data. This can lead to poor performance on the test data.

Overall, dimensionality reduction can be a powerful tool for data analysis. However, it is important to be aware of the potential drawbacks of these methods before using them.

### Loss Function Techniques

Here are some of the most common loss functions used in machine learning:

* **Mean squared error (MSE):** This is the most common loss function for regression problems. It measures the squared difference between the predicted values and the actual values.
* **Mean absolute error (MAE):** This is another common loss function for regression problems. It measures the absolute difference between the predicted values and the actual values.
* **Cross-entropy loss:** This is the most common loss function for classification problems. It measures the difference between the predicted probabilities and the actual labels.
* **Hinge loss:** This is a loss function that is often used for support vector machines. It measures the distance between the decision boundary and the data points.
* **Huber loss:** This is a loss function that is less sensitive to outliers than MSE. It is often used for regression problems where there may be outliers in the data.
* **Logistic loss:** This is a loss function that is often used for logistic regression. It measures the log-likelihood of the data given the model parameters.

The best loss function to use for a particular problem will depend on the nature of the data and the goals of the model. For example, MSE is often a good choice for regression problems where the data is normally distributed. Cross-entropy loss is often a good choice for classification problems where the classes are well-separated.

It is important to note that loss functions are not always used to train machine learning models. In some cases, they can be used to evaluate the performance of a model after it has been trained. For example, you might use MSE to evaluate the accuracy of a regression model.

### Hyperparameter Tuning Approaches
1. Grid Search 
2. Random Search
3. Bayesian Optimization

### Machine learning signal pre-processing techniques
We will be using the following pre-processing techniques:

1. Fast Fourier Transform
2. Discrete Wavelet Transform
3. Calculate Feature Vectors by Simpsons Rule.
Deep learning will not require pre-processing since it is meant to learn features by itself, we will however experiment and see the results.
### Machine Learning Approaches

We will be surveying the following algorithms in the is project:

1. Logistic Regression
2. Naive Bayes
3. K-Nearest Neighbors
4. Decision Tree
5. Support Vector Machines
6. Random Forest

### Deep Learning Approaches

We will be leveragin Supervised learning tasks:

![Deep learning Architectures](./img/Screenshot%202023-06-01%20230540.png)

We will be using CNNs for images primarily

## Goals and Milestones

- [x] Download the data
- [ ] Summarise understanding of the data like the distribution and probabilites and other statical information
- [ ] Pre-process the data
- [ ] Train the data to create a machine learning model
- [ ] Test the machine learning model with unseen data
- [ ] Train the data to creat a deep learning model
- [ ] Test the deep learning model with unseen data
- [ ] Containerise Model for use with streamlit for easy future testing.

## Questions

| Question     | Answers |
| ----------- | ----------- |
| What is suppose to be the buffer size of the signal that we take?     |        |
| How many classes do we consider?          |       Refer [this](./references/Arrhythmia_Detection_-_A_Machine_Learning_based_Comparative_Analysis_with_MIT-BIH_ECG_Data.pdf) |
|Do we always extract the QRS complex? | |
|What are we looking for in an ECG Data?|
|Do we Normalize or standardise the data the data?|| 


## Resources 

- https://realpython.com/python-scipy-fft/
- https://swharden.com/blog/2020-09-23-signal-filtering-in-python/
- https://danielmuellerkomorowska.com/2020/06/02/smoothing-data-by-rolling-average-with-numpy/
- https://refactored.ai/microcourse/notebook?path=content%2F06-Classification_models_in_Machine_Learning%2F02-Multivariate_Logistic_Regression%2Fmulticlass_logistic-regression.ipynb
- https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/

## Processed Data

https://drive.google.com/file/d/1ANq9kiWBsPsQpmsfJ-v4fpOMn0lEKypv/view?usp=sharing

