# Arrhythmia Detection

## Introduction

We will be using the [MIT data](https://archive.physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm) to train a model to detect Arrhythmias.

## Summary of the data after Analyzing.

## Type of problem

1. Classification problem
2. Supervised Learning problem.

## Tools And Frameworks
- [WFDB](https://wfdb.readthedocs.io/en/latest/)
- [Librosa](https://librosa.org/doc/latest/index.html)
- [Pytorch](https://pytorch.org/)
- [Pandas](https://pandas.pydata.org/)
- [Seaborn](https://seaborn.pydata.org/)

## Approaches

### Dimensionality Reduction Techniques
1. Principal Component Analysis
### Hyperparameter Tuning Approaches
1. Grid Search 
2. Random Search
3. Bayesian Optimization

### Machine learning signal pre-processing techniques
We will be using the following pre-processing techniques:

1. Fast Fourier Transform
2. Discrete Wavelet Transform

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
1. What is suppose to be the buffer size of the signal that we take?
2. How many classes do we consider?

