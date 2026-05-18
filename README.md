# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries: Pandas, Matplotlib, SVM modules, and TF-IDF vectorizer from scikit-learn.

2.Load the spam dataset, select required columns, and convert labels (ham = 0, spam = 1) into numeric values.

3.Extract message text as X and labels as Y, then convert text data into numerical form using TF-IDF vectorization.

4.Split the dataset into training and testing sets.

5.Create and train the SVM model using the training data.

6.Predict test results, generate the confusion matrix, and display it using a heatmap. 

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: VIGNESH.K
RegisterNumber:  212225240183

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("spam (1).csv", encoding='latin-1')

data = data[['v1','v2']]
data.columns = ['label','message']
data['label'] = data['label'].map({'ham':0, 'spam':1})
x = data['message']
y = data['label']
vectorizer = TfidfVectorizer(stop_words='english')
x_vectorized = vectorizer.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_vectorized, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',xticklabels=['Ham','Spam'],yticklabels=['Ham','Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for SVM Spam Detection")
plt.show()
```

## Output:
<img width="617" height="492" alt="image" src="https://github.com/user-attachments/assets/7f61117a-3c3c-44aa-8ed3-4a1ac37123b1" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

