import librosa
import numpy as np
import pandas as pd
import pathlib
import os
import csv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import classifiers as cl
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


data = pd.read_csv('data2.csv')
data.head()
data.shape
data = data.drop(['label'],axis=1)


name = data.iloc[:, 0]
encoder = LabelEncoder()
y = encoder.fit_transform(name)
scaler = StandardScaler()
what = data.iloc[:, :-1]
example = np.array(data.iloc[:, 1:], dtype = float)
X = scaler.fit_transform(example)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001)
names = name.tolist()


y_pred,pred = cl.pred_Neural_network(X_train, y_train,X_test,names)
print('For track:')
print(names[int(y_test)])
print('Most similar song is:')
print(y_pred[0])
print('With a similarity of a:')
print(pred[0])
