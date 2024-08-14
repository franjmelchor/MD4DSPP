#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np


# In[32]:


data_model = pd.read_csv('./python_dataDictionaries/numericBinner_output_dataDictionary.csv',sep=',')
data_model.head()


# In[33]:


data_model= data_model.drop(columns=data_model.columns[0:8])


# **Partitioning**

# In[34]:


data_model.shape


# In[35]:


from sklearn.model_selection import train_test_split
X = pd.get_dummies(data_model.dropna().drop(['Enroll'],axis=1))
y = data_model.dropna()['Enroll']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.3,
                                                    random_state=53190)


# In[36]:


data_model_train_data = X_train.copy()
data_model_train_data['Enroll'] = y_train
data_model_train_data.shape


# In[37]:


data_model_test_data = X_test.copy()
data_model_test_data['Enroll'] = y_test
data_model_test_data.shape


# **Equal Size Sampling**

# In[38]:


data_model_train_data['Enroll'].value_counts()


# In[39]:


class_0 = data_model_train_data[data_model_train_data['Enroll']==0]
class_1 = data_model_train_data[data_model_train_data['Enroll']==1]


# In[40]:


class_1 = class_1.sample(len(class_0))


# In[41]:


data_model_train_data = pd.concat([class_0,class_1],axis=0)


# **Random Forest Learner**

# In[42]:


from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(random_state=1634694125)
clf_rf.fit(data_model_train_data.drop(['Enroll'],axis=1), data_model_train_data['Enroll'])


# **Random Forest Predictor**

# In[43]:


X_test['Prediction ()'] = clf_rf.predict(data_model_test_data.drop(['Enroll'],axis=1))


# **Scorer**

# In[44]:


from sklearn.metrics import confusion_matrix
y_pred = X_test['Prediction ()']
y_true = data_model_test_data['Enroll']
confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
print(confusion_matrix)


# In[45]:


from sklearn.metrics import accuracy_score
accuracy_score(y_true=y_true, y_pred=y_pred)*100


# In[46]:


from sklearn.metrics import cohen_kappa_score
cohen_kappa_score(y_pred, y_true)


# In[47]:


wrong_classified = confusion_matrix[0,1] + confusion_matrix[1,0]
correct_classified = confusion_matrix[0,0] + confusion_matrix[1,1]

error = (wrong_classified/(wrong_classified + correct_classified))*100
error


# **Roc Curve**

# In[48]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_true=y_true,y_score=y_pred)


# In[ ]:





# In[ ]:




