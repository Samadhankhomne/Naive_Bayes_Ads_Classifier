# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\91976\OneDrive\Desktop\NIT\October_2024\15th - SVM\Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import Normalizer
sc = Normalizer()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Training the Naive Bayes model on the Training set

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True) 
classifier.fit(X_train, y_train)

'''
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB() 
classifier.fit(X_train, y_train)


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB() 
classifier.fit(X_train, y_train)
'''

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:\n",cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("accuracy score:",ac)

bias = classifier.score(X_train, y_train)
print("bias:",bias)

variance = classifier.score(X_test, y_test)
print("variance:",variance)

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print("classification Rreport:\n",cr)