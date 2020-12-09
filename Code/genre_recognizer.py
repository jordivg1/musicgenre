import librosa
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import classifiers as cl
import data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

X,y = data.get_data('data.csv')
X2,y2 = data.get_data('data2.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2)


#X_test = X_train[:200]
#y_test = y_train[:200]
#X_train = X_train[400:]
#y_train = y_train[400:]


y_true = []
y_true2 = []

for i in range(len(y_test)):
    label = y_test[i]
    y_true.append(genres[label])
    label2 = y_test2[i]
    y_true2.append(genres[label2])

#y_pred = cl.Neural_network(X_train, y_train, X_test,genres)
#y_pred2 = cl.Neural_network(X_train2, y_train2, X_test2,genres)

#y_pred = cl.SVM(X_train, y_train,X_test,genres)
#y_pred2 = cl.SVM(X_train2, y_train2,X_test2,genres)

#y_pred = cl.Gaussian(X_train,y_train,X_test,genres)
#y_pred2 = cl.Gaussian(X_train2, y_train2,X_test2,genres)

#y_pred = cl.Tree(X_train,y_train,X_test,genres)
#y_pred2 = cl.Tree(X_train2, y_train2,X_test2,genres)

y_pred = cl.RandomForest(X_train,y_train,X_test,genres)
y_pred2 = cl.RandomForest(X_train2, y_train2,X_test2,genres)

#y_pred = cl.MLP(X_train,y_train,X_test,genres)




print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred,digits = 3))
print(confusion_matrix(y_true2, y_pred2))
print(classification_report(y_true2, y_pred2,digits = 3))

#plt.plot(y_pred, y_true, '.')
#plt.show()
