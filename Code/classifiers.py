import librosa
import pandas as pd
import numpy as np
import keras
from keras import models
from keras import layers
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

def Neural_network(X_train, y_train,X_test,genres):
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(256, activation='relu'))
    #model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train,y_train,epochs=20,batch_size=128)
    predictions = model.predict(X_test)
    y_pred = []
    for i in range(len(predictions)):
        label = np.argmax(predictions[i])
        y_pred.append(genres[label])
    return y_pred

def SVM(X_train,y_train,X_test,genres):
    svm = SVC(class_weight='balanced', gamma='scale')
    svm.fit(X_train,y_train)
    predictions = svm.predict(X_test)
    y_pred = []
    for i in range(len(predictions)):
        label = predictions[i]
        y_pred.append(genres[label])
    return y_pred

def Gaussian(X_train,y_train,X_test,genres):
    GNB = GaussianNB()
    GNB.fit(X_train,y_train)
    predictions = GNB.predict(X_test)
    y_pred = []
    for i in range(len(predictions)):
        label = predictions[i]
        y_pred.append(genres[label])
    return y_pred

def Tree(X_train,y_train,X_test,genres):
    DTC = DecisionTreeClassifier()
    DTC.fit(X_train,y_train)
    predictions = DTC.predict(X_test)
    y_pred = []
    for i in range(len(predictions)):
        label = predictions[i]
        y_pred.append(genres[label])
    return y_pred

def RandomForest(X_train,y_train,X_test,genres):
    RFC = RandomForestClassifier(random_state=42, n_jobs=4,class_weight='balanced',n_estimators=250,bootstrap=True)
    RFC.fit(X_train,y_train)
    predictions = RFC.predict(X_test)
    y_pred = []
    for i in range(len(predictions)):
        label = predictions[i]
        y_pred.append(genres[label])
    return y_pred

def MLP(X_train,y_train,X_test,genres):
    MLP = MLPClassifier()
    MLP.fit(X_train,y_train)
    predictions = MLP.predict(X_test)
    y_pred = []
    for i in range(len(predictions)):
        label = predictions[i]
        y_pred.append(genres[label])
    return y_pred

def pred_Neural_network(X_train, y_train,X_test,names):
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(256, activation='relu'))
    #model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1000, activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train,y_train,epochs=20,batch_size=128)
    predictions = model.predict(X_test)
    y_pred = []
    pred = []
    #sorted = np.sort(predictions)
    for i in range(len(predictions)):
        label = np.argmax(predictions[i])
        pred.append(max(predictions[i])*100)
        y_pred.append(names[label])
    return y_pred,pred
