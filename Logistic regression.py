#!/usr/bin/env python
# coding: utf-8

# # Automatic File Sorter in File Explorer
# 

# In[1]:


#import libraries


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


#load data


# In[5]:


td=pd.read_csv('titanic_train.csv')


# In[6]:


len(td)


# In[7]:


td.head()


# In[8]:


td.index


# In[9]:


td.columns


# In[10]:


td.info()


# In[11]:


td.describe()


# In[12]:


#countplot of subrvived vs not  survived


# In[13]:


sns.countplot(x='Survived',data=td)


# In[14]:


#Male vs Female Survived?


# In[15]:


sns.countplot(x='Survived',data=td,hue='Sex')


# In[16]:


#Check for null


# In[17]:


td.isna()


# In[18]:


#Check how many values are null


# In[19]:


td.isna().sum()


# In[20]:


#Visualize null values


# In[21]:


sns.heatmap(td.isna())


# In[22]:


#find the % of null values in age column


# In[23]:


(td['Age'].isna().sum()/len(td['Age']))*100


# In[24]:


#find the % of null values in cabin column
(td['Cabin'].isna().sum()/len(td['Cabin']))*100


# In[25]:


#find the distribution for the age column
sns.displot(x='Age',data=td)


# In[26]:


#Data Cleaning
#fill age column


# In[28]:


td['Age'].fillna(td['Age'].mean(),inplace=True)


# In[29]:


#verify null value
td['Age'].isna().sum()


# In[30]:


#visualize null values
sns.heatmap(td.isna())


# In[31]:


#Drop cabin column
td.drop('Cabin',axis=1,inplace=True)


# In[32]:


#see the contents of the data
td.head()


# In[33]:


#Check for the non-numeric column
td.info()


# In[34]:


#convert sex column to numerical values
gender=pd.get_dummies(td['Sex'],drop_first=True)


# In[35]:


td['Gender']=gender


# In[36]:


td.head()


# In[37]:


#drop the columns which are not required
td.drop(['Name','Sex','Ticket','Embarked'],axis=1,inplace=True)


# In[38]:


td.head()


# In[39]:


#Seperate Dependent and Independent variables
x=td[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Gender']]
y=td['Survived']


# In[40]:


y


# In[41]:


#Build the model
#import train test split method


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[44]:


#import Logistic  Regression
from sklearn.linear_model import LogisticRegression


# In[47]:


#Fit  Logistic Regression 
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[48]:


#predict
predict=lr.predict(x_test)


# In[49]:


#Testing
#print confusion matrix 


# In[50]:


from sklearn.metrics import confusion_matrix


# In[51]:


pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted No','Predicted Yes'],index=['Actual No','Actual Yes'])


# In[53]:


#import classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




