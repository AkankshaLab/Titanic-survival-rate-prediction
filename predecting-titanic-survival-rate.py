#!/usr/bin/env python
# coding: utf-8

# In[40]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


df = pd.read_csv("train.csv")


# In[3]:


df.head()


# In[5]:


df.shape


# In[6]:


df.isna().sum()


# In[8]:


df.Sex.value_counts()


# In[9]:


female_percent = 314 / (314 + 577)
male_percent = 577 / (314 + 577)


# In[11]:


female_percent * 100 , male_percent * 100


# In[17]:


female_survivors = df[df['Survived'] == 1][df['Sex'] == 'female']
male_survivors = df[df['Survived'] == 1][df['Sex'] == 'male']


# In[18]:


female_survivors


# In[24]:


len(female_survivors), len(male_survivors)


# In[31]:


df.Sex.value_counts().plot(kind = 'bar');


# In[33]:


df.Survived.value_counts().plot(kind = 'bar');


# In[22]:


pd.crosstab(df.Survived, df.Sex).plot(kind='bar');


# In[36]:


df.columns


# In[39]:


df.Pclass.value_counts().plot(kind='bar');


# In[35]:


pd.crosstab(df.Survived, df.Pclass).plot(kind='bar');


# In[44]:


np.mean(df.Age)


# In[46]:


df.Age[:100]


# In[48]:


np.median(df.Age)


# In[49]:


df.Age.fillna(np.mean(df.Age), inplace=True)


# In[50]:


df.Age[:100]


# In[51]:


df.Age.isna().sum()


# In[53]:


df.Age.hist();


# In[54]:


for label, content in df.items():
    if not pd.api.types.is_numeric_dtype(content):
        df[label] = pd.Categorical(content).codes+1


# In[55]:


for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                df[label] = content.fillna(content.median())


# In[56]:


df.isna().sum()


# In[65]:


from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

X_train = df.drop("Survived", axis=1)
y_train = df["Survived"]

clf = RandomForestClassifier()

clf.fit(X_train, y_train)


# In[66]:


from sklearn.model_selection import train_test_split

np.random.seed(42)

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
clf.score(X_val, y_val)


# In[59]:


df_test = pd.read_csv("test.csv")


# In[62]:


df_test.Age.fillna(np.mean(df_test.Age), inplace=True)
for label, content in df_test.items():
    if not pd.api.types.is_numeric_dtype(content):
        df_test[label] = pd.Categorical(content).codes+1
for label, content in df_test.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                df_test[label] = content.fillna(content.median())


# In[63]:


y_preds = clf.predict(df_test)


# In[67]:


y_preds


# In[68]:


df.columns


# In[69]:


df_preds = pd.DataFrame()
df_preds["PassengerId"] = df_test["PassengerId"]
df_preds["Survived"] = y_preds
df_preds


# In[70]:


df_preds.to_csv("Survival_predictions.csv", index=False)


# In[71]:


from IPython.display import FileLink
FileLink("Survival_predictions.csv")


# In[ ]:





# In[ ]:




