#!/usr/bin/env python
# coding: utf-8

# ## TOPIC: Wine Quality Prediction 

# Machine Learning model to predict the
# quality of wine using linear regression
# only Jupyter notebook code.

# ## INTRODUCTION

# In this Jupyter Notebook assignment, we will explore the world of wine quality prediction using machine learning. Wine quality is influenced by various chemical properties, and we will use a dataset containing these properties to build a predictive model.Wine quality is typically assessed on a scale, and we will aim to predict this numerical quality score.

# ### ABOUT THE DATASET

# We will be using the "Wine Quality" dataset. This dataset contains a range of attributes such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, and wine quality.

# ### STEPS

# To solve this problem, we will follow these steps:
# 
# 1.Data Preparation: We will load the dataset, perform any necessary data cleaning, and explore the dataset to understand its characteristics.
# 
# 2.Data Splitting: We will split the dataset into a training set and a testing set to evaluate the performance of our model.
# 
# 3.Linear Regression: We will build a linear regression model using scikit-learn, a popular machine learning library in Python.
# 
# 4.Model Training: We will train the model on the training dataset, allowing it to learn the relationships between the input features (chemical properties) and the target variable (wine quality).
# 
# 5.Model Evaluation: We will evaluate the model's performance using metrics such as Mean Squared Error (MSE) and R-squared (R2) score.
# 
# 6.Visualization: We will create visualizations to better understand the model's predictions and compare them to the actual wine quality.

# In[1]:


#Importing required packages.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


wine = pd.read_csv("C:\winequality-red.csv")
wine


# In[4]:


#Let's check how the data is distributed
wine.head()


# In[5]:


#Information about the data columns
wine.info()


# #### Let's do some plotting to know how the data columns are distributed in the dataset

# In[6]:


#Here we see that fixed acidity does not give any specification to classify the quality.
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)


# Here we see that fixed acidity does not give any specification to classify the quality

# In[8]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = wine)


# Composition of citric acid go higher as we go higher in the quality of the wine

# In[7]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)


# Here we see that its quite a downing trend in the volatile acidity as we go higher the quality 

# In[9]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = wine)


# In[10]:


#Composition of chloride also go down as we go higher in the quality of the wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = wine)


# Composition of chloride also go down as we go higher in the quality of the wine

# In[11]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine)


# In[13]:


#Sulphates level goes higher with the quality of wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = wine)


# Sulphates level goes higher with the quality of wine

# In[14]:


#Alcohol level also goes higher as te quality of wine increases
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = wine)


# Alcohol level also goes higher as te quality of wine increases

# ### Linear Regression 

# In[36]:


X = wine.drop("quality", axis=1)
y = wine["quality"]


# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[41]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)


# In[42]:


# Calculate Mean Squared Error (MSE) and R-squared (R2) score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
r2


# **Interpretation** An R2 score of approximately 0.4032 means that the model explains about 40.32% of the variability in wine quality based on the independent variables in your model. This suggests that some of the characteristics or features used in the regression model (e.g., acidity, alcohol content, pH, etc.) do have an influence on wine quality, but there are other factors or complexities that are not captured by your model.

# In[44]:


print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")


# **Interpretation** A mean suqared error of 0.38 suggests that your model is making relatively small prediction errors, which is generally a positive sign. A low MSE value like this indicates that the model is performing reasonably well in terms of minimizing prediction errors for wine quality. 

# ### Preprocessing Data for performing Machine learning algorithm

# In[15]:


#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)


# In[16]:


#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()


# In[17]:


#Bad becomes 0 and good becomes 1 
wine['quality'] = label_quality.fit_transform(wine['quality'])


# In[18]:


wine['quality'].value_counts()


# In[19]:


sns.countplot(wine['quality'])


# In[20]:


#Now seperate the dataset as response variable and feature variabes
X = wine.drop('quality', axis = 1)
y = wine['quality']


# In[21]:


#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[22]:


#Applying Standard scaling to get optimized result
sc = StandardScaler()


# In[23]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# **Random Forest Classifier**
# A "Random Forest Classifier" is a machine learning algorithm used for classification tasks. It is an ensemble learning method that combines the predictions of multiple decision trees to make accurate and robust predictions. 
# 
# 

# In[24]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# In[25]:


#Let's see how our model performed
print(classification_report(y_test, pred_rfc))


# **Interpretation** 
# 
# This classification report provides insights into how well the model performs for each class and overall. For class 0, the model has higher precision, recall, and F1-score, indicating better performance compared to class 1. The weighted average values give an overall assessment of the model's performance, considering class imbalances. The model appears to perform better at classifying instances of class 0 than class 1.

# In[26]:


#Confusion matrix for the random forest classification
print(confusion_matrix(y_test, pred_rfc))


# The model correctly predicted class 0 (negative class) for 262 instances (TN), indicating good performance in identifying true negatives.
# 
# The model incorrectly predicted class 1 (positive class) for 11 instances (FP), which are false positives.
# 
# The model incorrectly predicted class 0 (negative class) for 28 instances (FN), which are false negatives.
# 
# The model correctly predicted class 1 (positive class) for 19 instances (TP), indicating good performance in identifying true positives.

# ## CONCLUSION

# In conclusion, we successfully built a linear regression model to predict the quality of wine based on relevant features. The model demonstrated a reasonable level of predictive accuracy, as evidenced by our evaluation metrics.This Jupyter notebook provides a solid foundation for predicting wine quality using linear regression, and it can serve as a starting point for further exploration and refinement of the model.
