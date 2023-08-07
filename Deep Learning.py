#!/usr/bin/env python
# coding: utf-8

# # Deep Learning

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading data 

# In[3]:


df = pd.read_csv("./processed_data/processed_data/ecg_processed_data.csv", index_col=0)


# In[4]:


df.head()


# In[5]:


df.describe().transpose()


# In[6]:


df.info()


# ## Preprocessing

# In[7]:


X = df[[str(i) for i in range(0, 200)]].values


# In[8]:


X


# In[9]:


X.shape


# In[10]:


y = df["Annotation Class"].astype("category")


# In[11]:


y.describe()


# In[12]:


y.info()


# In[13]:


y.cat.codes


# In[14]:


y.cat.categories


# In[15]:


y_code = y.cat.codes.values


# In[16]:


y_code


# In[17]:


y_code.shape


# In[18]:


y_code


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y_code, test_size=0.2, random_state=101, stratify=y, shuffle=True
)


# In[21]:


X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=101, stratify=y_train, shuffle=True
)


# In[22]:


X_train_tensor = torch.FloatTensor(X_train)
X_val_tensor = torch.FloatTensor(X_val)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train)
y_val_tensor = torch.LongTensor(y_val)
y_test_tensor = torch.LongTensor(y_test)


# In[23]:


X_train_tensor.shape


# In[24]:


X_val_tensor.shape


# ## ANN

# In[25]:


from ecg_deep_learning_models.models import ECGANNModel, ECGCNNModel, ECGLSTMModel
from ecg_deep_learning_models.utils import (
    count_parameters,
    train_model,
    eval_model,
    show_metrics,profile
)


# In[26]:


ecg_ann_model = ECGANNModel(layers=[100, 50])
ecg_ann_model_1 = ECGANNModel(layers=[100, 50, 25])


# In[27]:


count_parameters(ecg_ann_model)


# In[28]:


count_parameters(ecg_ann_model_1)


# In[29]:


ecg_ann_model


# In[30]:


ecg_ann_model_1


# In[31]:


X_train_tensor.shape


# In[32]:


ecg_ann_model.parameters()


# In[33]:


ecg_ann_model_1.parameters()


# In[34]:


ecg_ann_model.parameters


# In[35]:


ecg_ann_model_1.parameters


# In[36]:


train_accuracies, test_accuracies, train_mean_losses, test_mean_losses = train_model(
    model=ecg_ann_model,
    X_train=X_train_tensor,
    y_train=y_train_tensor,
    X_test=X_val_tensor,
    y_test=y_val_tensor,
    learning_rate=0.001,
    batch_size=100,
    val_batch_size=100,
    epochs=100,
)


# In[38]:


show_metrics(train_accuracies, test_accuracies, train_mean_losses, test_mean_losses)
eval_model(ecg_ann_model, X_test_tensor, y_test_tensor)
profile(ecg_ann_model,tuple([200]))


# In[39]:


train_accuracies, test_accuracies, train_mean_losses, test_mean_losses = train_model(
    model=ecg_ann_model_1,
    X_train=X_train_tensor,
    y_train=y_train_tensor,
    X_test=X_val_tensor,
    y_test=y_val_tensor,
    learning_rate=0.001,
    batch_size=100,
    val_batch_size=100,
    epochs=100,
)


# In[40]:


show_metrics(train_accuracies, test_accuracies, train_mean_losses, test_mean_losses)
eval_model(ecg_ann_model_1, X_test_tensor, y_test_tensor)
profile(ecg_ann_model_1,tuple([200]))


# ## CNN

# In[41]:


ecg_cnn_model = ECGCNNModel()


# In[42]:


count_parameters(ecg_cnn_model)


# In[43]:


train_accuracies, test_accuracies, train_mean_losses, test_mean_losses = train_model(
    model=ecg_cnn_model,
    X_train=X_train_tensor.reshape(-1, 1, 200),
    y_train=y_train_tensor,
    X_test=X_val_tensor.reshape(-1, 1, 200),
    y_test=y_val_tensor,
    learning_rate=0.001,
    batch_size=100,
    val_batch_size=100,
    epochs=6,
)


# In[46]:


show_metrics(train_accuracies, test_accuracies, train_mean_losses, test_mean_losses)
eval_model(ecg_cnn_model, X_test_tensor.reshape(-1, 1, 200), y_test_tensor)
profile(ecg_cnn_model, tuple([1,200]))


# ## LSTM

# In[47]:


ecg_lstm_model = ECGLSTMModel()


# In[48]:


ecg_lstm_model.parameters()


# In[49]:


ecg_lstm_model.parameters


# In[50]:


count_parameters(ecg_lstm_model)


# In[51]:


train_accuracies, test_accuracies, train_mean_losses, test_mean_losses = train_model(
    model=ecg_lstm_model,
    X_train=X_train_tensor.reshape(-1,1,200),
    y_train=y_train_tensor,
    X_test=X_val_tensor.reshape(-1,1,200),
    y_test=y_val_tensor,
    learning_rate=0.001,
    batch_size=100,
    val_batch_size=100,
    epochs=10, lstm=True
)


# In[52]:


show_metrics(train_accuracies, test_accuracies, train_mean_losses, test_mean_losses)
eval_model(ecg_lstm_model, X_test_tensor.reshape(-1, 1, 200), y_test_tensor)
profile(ecg_lstm_model,tuple([ 1, 200]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




