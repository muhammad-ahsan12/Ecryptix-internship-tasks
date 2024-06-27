# -*- coding: utf-8 -*-
"""Email Classifier.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1D5Run6Ru3u7r4B5Pxge7a-Wy6XCMlMF2
"""

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!kaggle datasets download -d uciml/sms-spam-collection-dataset

import zipfile
zip_ref = zipfile.ZipFile('/content/sms-spam-collection-dataset.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

# Importing necessary libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

email_data = pd.read_csv('/content/spam.csv', encoding='latin-1')
email_data.head()

email_data = email_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

email_data.columns = ['label', 'message']
email_data.head()

print(f"the shape of dataset : {email_data.shape}")
email_data.info()

email_data.value_counts('label')

sns.countplot(x='label', data=email_data)
plt.show()

"""# Separate the data into X and y"""

X = email_data['message']
y = email_data['label']

print(X)

print(y)

"""# Convert label columns into numarical data"""

labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(y)

print(y)

"""# Now convert the data in to training and testing part"""

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

"""# Clean the Data or Remove unnesessry words"""

def clean_email_body(body):
    body = re.sub(r'<[^>]+>', '', body)
    body = re.sub(r'\W', ' ', body)
    body = re.sub(r'\s+', ' ', body)
    return body.strip().lower()

# Clean the training and testing data
X_train = X_train.apply(clean_email_body)
X_test = X_test.apply(clean_email_body)

"""# Transform text data to TF-IDF features for classification."""

tf_id = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_tf = tf_id.fit_transform(X_train)
X_test_tf = tf_id.transform(X_test)

print(X_train_tf)

"""# Function for Plot Confusion Matricx"""

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

"""# Train Models"""

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(kernel='linear')
}

accuracy_scores = []
model_names = []
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_tf, y_train)
    y_pred = model.predict(X_test_tf)
    accuracy = model.score(X_test_tf, y_test)
    accuracy_scores.append(accuracy)
    model_names.append(name)
    print(f"Accuracy of {name}: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Confusion Matrix:")
    plot_confusion_matrix(cm, name)

"""# Accuracy of different Models"""

plt.figure(figsize=(7, 4))
plt.bar(model_names, accuracy_scores, color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.show()

"""# Function for checking wheather email is Spam or Ham"""

def send_mail(mail):
  extract_mail=tf_id.transform(mail)
  prediction=model.predict(extract_mail.toarray())
  return prediction

input_mail=["Romantic Paris. 2 nights, 2 flights from £79 Book now 4 next year. Call 08704439680Ts&Cs apply."]
prediction=send_mail(input_mail)
if prediction==1:
  print("the email is spam")
else:
  print("the email is ham")

