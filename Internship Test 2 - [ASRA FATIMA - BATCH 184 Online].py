#!/usr/bin/env python
# coding: utf-8

# In[135]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# <h1 style='color:blue'> INTRODUCTION </H1>

# About Data:
# - The data has two columns - input (feature) and output (target). 
# - The output is to be predicted based on a given input values.
# - The data is a labeled data.
# 
# Goal:
# - Perform proper analysis of the dataset and draw conclusions based on the analysis.
# - Build a Machine Learning Model to predict output based on the input column.

# <h1 style='color:blue'> IMPORT THE DATA </H1>

# ### Read the dataset

# In[3]:


df = pd.read_csv("d:\\Users\\NAME\\Desktop\\INNOMATICS\\QUIZZEZZ!!\\Internship Test 2\\dataframe_.csv")


# In[4]:


df


# In[5]:


# Checking the number of rows and columns in the datset
df.shape


# In[7]:


# Finding a brief summary of the dataset
df.info()


# In[8]:


# Finding the statistical summary of the dataset
df.describe()


# In[9]:


np.round(df.describe(),2)


# <H1 style='color: blue'> DATA CLEANING  </H1>

# ### Checking for Missing values

# In[10]:


df.isnull().any()


# In[11]:


df.isnull()


# In[12]:


df.isnull().sum()


# In[17]:


df[df["input"].isnull()]


# - The above data shows that row with index 1439 has null values in both input and output

# In[22]:


# Checking values in the rows above and below the null values row
df.loc[1437:1441,]


# ##### Demonstrating different methods to fill missing values
# ##### Fill missing values with mean or median or mode

# In[30]:


# Filling missing values with mean

df['input']= df['input'].fillna(df['input'].mean())


# In[32]:


df['input'].isnull().any()


# In[33]:


df['output']= df['output'].fillna(df['output'].mean())


# In[34]:


df['output'].isnull().any()


# In[ ]:





# In[36]:


# Filling missing values with median

df= pd.read_csv("d:\\Users\\NAME\\Desktop\\INNOMATICS\\QUIZZEZZ!!\\Internship Test 2\\dataframe_.csv")


# In[38]:


df['input']= df['input'].fillna(df['input'].median())


# In[39]:


df['input'].isnull().any()


# In[41]:


df['output']= df['output'].fillna(df['output'].median())


# In[42]:


df['output'].isnull().any()


# In[ ]:





# In[43]:


# Filling missing values with mode

df= pd.read_csv("d:\\Users\\NAME\\Desktop\\INNOMATICS\\QUIZZEZZ!!\\Internship Test 2\\dataframe_.csv")


# In[54]:


df['input']= df['input'].fillna(df['input'].mode()[0])


# In[55]:


df['input'].isnull().any()


# In[56]:


df['output']= df['output'].fillna(df['output'].mode()[0])


# In[57]:


df['output'].isnull().any()


# In[ ]:





# In[61]:


# Dropping the missing value row

df= pd.read_csv("d:\\Users\\NAME\\Desktop\\INNOMATICS\\QUIZZEZZ!!\\Internship Test 2\\dataframe_.csv")
df.dropna()


# - The above shows that 1 row has been deleted. And now we have only 1696 rows.

# In[63]:


df.describe()


# - the mean and median are the same values after dropping the null values

# In[64]:


# Using the dataset after dropping the null values


# In[67]:


df['input'].value_counts()


# - There are 991 unique input values in the datset given

# In[69]:


df["output"].value_counts()


# - There are 969 unique output values in the given datset

# - From above we can conclude that there are 22 input values which have same output.

# In[ ]:





# <H1 style= 'color: blue'>Exploratory Data Analysis: </H1>
# 
# - **Univariate Analysis (Analysis for single column)**
# - **Bi-variate Analysis (Analysis about two columns)**
# - **Multi-variate Analysis (Analysis about more the than two columns)**

# #### - **Univariate Analysis (Analysis for single column)**
# - Analysing single columns separately.
# - Trying to see their distribution and getting insights from them.
# - Identifying and treating outliers

# In[73]:


# Checking the distribution of input data
sns.distplot(df['input'])


# In[74]:


sns.boxplot(df['input'])


# - Majority of the input values are in range -50 to +50.

# In[75]:


df['input'].describe()


# - The above summary shows that the 
#  - minimum value input is : -134.962839
#  - maximum value input is :  134.605775 

# In[78]:


# Checking the distribution of output data
sns.distplot(df['output'])


# In[80]:


sns.boxplot(df['output'])


# - The outliers are concentrated above 100
# - Majority of the output values are negative

# In[79]:


df['output'].describe()


# - The above summary shows that the 
#  - minimum value output is : -132.422167
#  - maximum value output is :  134.425495 

# In[81]:


df


# #### Handling Outliers

# In[103]:


def find_outliers(dataset, column, verbose = False):
    q1 = dataset[column].quantile(0.25)
    q3 = dataset[column].quantile(0.75)
    IQR = q3 - q1
    lower_boundary = q1 - (1.5 * IQR)
    upper_boundary = q3 + (1.5 * IQR)
    print(upper_boundary)
    outliers = dataset[(dataset[column] < lower_boundary) | (dataset[column] > upper_boundary)]
    if verbose:
        display(outliers)
        print("Outliers Percentage :: ", (len(outliers) / len(dataset)) * 100, "%")
    return outliers


# In[105]:


find_outliers(df, 'input', verbose = True)


# - There are no outliers in the input data

# In[106]:


find_outliers(df, 'output', verbose = True)


# In[107]:


q1 = df['input'].quantile(0.25)
q3 = df['input'].quantile(0.75)
IQR = q3 - q1
lb = q1 - (1.5 * IQR)
ub = q3 + (1.5 * IQR)

df['input']= np.where(df['input'] > ub, ub, df['input'])


# In[108]:


df['input']= np.where(df['input'] < lb, lb, df['input'])


# In[109]:


sns.distplot(df['input'])


# In[110]:


find_outliers(df, 'input', verbose = True)


# In[111]:


q11 = df['output'].quantile(0.25)
q33 = df['output'].quantile(0.75)
IQR = q33 - q11
lbb = q11 - (1.5 * IQR)
ubb = q33 + (1.5 * IQR)

df['output']= np.where(df['output'] > ubb, ubb, df['output'])


# In[112]:


df['output']= np.where(df['output'] < lbb, lbb, df['output'])


# In[113]:


find_outliers(df, 'output', verbose = True)


# - Thus outliers are removed

# ###  Creating interactive plots

# In[114]:


from ipywidgets import interact, interact_manual


# - **Using the interact and interact_manual packages we are creating interactive plots as follows:**

# In[115]:


@interact_manual
def hist_plot(col = df.columns):
    sns.histplot(df[col])


# In[116]:


@interact
def boxplot(col=df.columns):
    sns.boxplot(df[col])


# In[143]:


@interact
def violinplot(col=df.columns):
    sns.violinplot(df[col])


# #### - **Bivariate Analysis (Analysis of two columns)**
# - Analysing two columns together.
# - Trying to see how they vary and getting insights from them.

# In[119]:


sns.scatterplot(df['input'], df['output'])


# - The above scater plot shows that the input and output values are highly related

# In[121]:


df.corr()


# - ***Demonstrating Binning***

# In[123]:


pd.cut(df['input'],[-134,-50,0,50,134]).value_counts()


# In[124]:


sns.countplot(pd.cut(df['input'],3, labels=['low','medium','high']).value_counts())


# In[125]:


sns.countplot(pd.cut(df['output'],3, labels=['low','medium','high']).value_counts())


# <H1 style= 'color: blue'> DESCRIPTIVE STATISTICS </H1>

# ### Measures of Central Tendency
#    - Mean
#    - Median
#    - Mode

# ### Measures of Dispersion
#    - Range (Max-Min values)
#    - Standard Deviation (Root mean squared error)
#    - Variance (mean squared error)

# In[126]:


df.describe()


# ### Measure of Shape
#    - SKEWNESS:
#          1. Positive Skewness:  [ mean > median ;  skew > 0.5 ] 
#              
#          2. Negative Skewness:  [ mean < median ;  skew < -0.5 ]
#              
#          3. Symmetric Skewness: [ mean ~= median ; -0.5 < skew < 0.5 ] (Normal Curve)

# In[128]:


df['input'].skew()


# - Negative slewness pattern is followed by input data

# In[129]:


sns.distplot(df['input'])


# In[130]:


df['output'].skew()


# - Positive skewness is followed by output data

# In[131]:


sns.distplot(df['input'])


# ### Measures of relation
# 
# #### Correlation:
#    - Relationship b/w two variables is called correlation.
#    - Types of correlation:
#        1. Possitive Correlation (correlation is in b/w 0.5 to 1)     [0.5 < corr() < 1]
#        2. Negative Correlation  (correlation is in b/w -0.5 to -1)   [-0.5 < corr() < -1]
#        3. No relation           (correlation is in b/w -0.5 to 0.5)  [-0.5 < corr() < 0.5]

# In[136]:


plt.figure(figsize=(8,7))
sns.set_palette("tab10")

sns.heatmap(df.corr(),annot=True,cmap="PiYG")


# In[137]:


df.corr()


# - We can conclude that the input and output values are positively correlated.

# <H1 style= 'color: blue'> DESCRIPTIVE STATISTICS </H1>

# ##### Split Dependent and Independent variables

# In[138]:


X= df['input']


# In[139]:


X   #---> Feature


# In[141]:


y = df['output']


# In[142]:


y


# ##### Split dataset into Train and Test

# In[145]:


from sklearn.model_selection import train_test_split


# In[146]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)


# In[147]:


X_train


# In[148]:


y_train


# #### Feature Scaling

# In[149]:


from sklearn.preprocessing import MinMaxScaler


# In[150]:


#create object of MinMaxScaler
sc = MinMaxScaler()


# In[152]:


pd.DataFrame(sc.fit_transform(X_train))


# In[153]:


from sklearn.preprocessing import StandardScaler


# In[154]:


sc = StandardScaler()


# In[158]:


#pd.DataFrame(sc.fit_transform(X_train))


# <H2 style= 'color: blue'> Observation: </H2>

# - The given data is labeled data, so a Supervised Learning Model should be used
# - Since the data is numeric, Regression algorithms should be used to train the model
# - We can use the KNN Algorithm for building the model to predict the output as it can find the nearest neighbor data points and predict the target from the given feature.

# In[ ]:




