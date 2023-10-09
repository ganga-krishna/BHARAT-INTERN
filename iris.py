#!/usr/bin/env python
# coding: utf-8

# # Iris Flowers Classification :

# ### TOPIC:
# Predict the different species of flowers on the length of there petals and sepals only Jupyter notebook code.

# To predict different species of flowers based on the length of their petals and sepals, you can use machine learning techniques, such as classification algorithms. One common dataset for this task is the Iris dataset, which contains measurements of petal and sepal lengths and widths for three different species of iris flowers: setosa, versicolor, and virginica.

# In[15]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[16]:


# Load the Iris dataset
iris = datasets.load_iris()


# In[17]:


# Features (sepal length, sepal width, petal length, petal width)
X = iris.data 
# Target (species)
y = iris.target 


# In[18]:


#Standardize the features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[19]:


#Train a machine learning model on the training data:
k = 3  # Number of neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)


# In[21]:


#Evaluate the model's performance:
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")


# The accuracy is 1.00, which means that the model correctly predicted the class labels for all 30 samples.

# In[24]:


#Make predictions on the test data:
y_pred = knn_classifier.predict(X_test)


# In[22]:


print("\nConfusion Matrix:")
print(confusion)

Actual Class 1 (Row 1): There are 10 samples in the dataset that truly belong to Class 1. The model correctly predicted all of them as Class 1 (True Positives).

Actual Class 2 (Row 2): There are 9 samples in the dataset that truly belong to Class 2. The model correctly predicted all of them as Class 2 (True Positives).

Actual Class 3 (Row 3): There are 11 samples in the dataset that truly belong to Class 3. The model correctly predicted all of them as Class 3 (True Positives).

In this case, the diagonal elements (from the top left to the bottom right) represent the true positives for each class. This means that the model correctly classified all the samples for each class, with no false positives or false negatives.

The confusion matrix indicates that the model performed perfectly for this dataset, correctly predicting all samples for each class.
# In[23]:


print("\nClassification Report:")
print(classification_rep)


# # INTERPRETATION

# For all three classes (0, 1, and 2), the precision is 1.00, which means that all positive predictions were correct. This indicates that the model does not produce any false positives for any class.
# 
# Three classes, the recall is 1.00, indicating that the model correctly identified all positive samples for each class.
# The F1-score is also 1.00 for all three classes, suggesting that the model achieves a perfect balance between precision and recall for each class.
# 
# Support represents the number of samples in each class. For Class 0, there are 10 samples; for Class 1, there are 9 samples; and for Class 2, there are 11 samples.
# 
# The macro-averaged precision, recall, and F1-score are all 1.00, indicating excellent performance across all classes.
#  
# Since the support for each class is roughly balanced in this case, the weighted-averaged metrics are also 1.00.

# # CONCLUSION

# In summary, the classification report confirms that the model has achieved perfect performance on this dataset. It correctly predicts all three classes with high precision and recall for each class, resulting in an overall accuracy of 1.00. 
