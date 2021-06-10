import xlwings as xw
import numpy as np
import pandas as pd
from sklearn import *
import sklearn
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras import initializers
import keras
import os

'''

Gör SVC/SVR
MLP
RNN
LSTM
CNN
XDGBoost

'''

def scale(X, XForecast):
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
    scaler.fit(X)
    X = scaler.transform(X)
    XForecast = scaler.transform(XForecast)

    return X, XForecast
    
    #return sklearn.preprocessing.normalize(X, norm='l2'), sklearn.preprocessing.normalize(XForecast, norm='l2')


def main():
    wb = xw.Book.caller()
    sheet = wb.sheets[0]
    if sheet["A1"].value == "Hello xlwings!":
        sheet["A1"].value = "Bye xlwings!"
    else:
        sheet["A1"].value = "Hello xlwings!"


@xw.func
def hello(name):
    return f"Hello {name}!"

@xw.func
@xw.arg("X", np.array, ndim = 2)
@xw.arg("Y", np.array)
@xw.arg("XForecast", np.array, ndim = 2)

def linreg(X, Y, XForecast, classification):

    if len(X.shape) == 1:
        X = X.reshape(-1,1)
        XForecast = XForecast.reshape(-1,1)
   
    X, XForecast = scale(X, XForecast)
    if classification:
        model = sklearn.linear_model.LogisticRegression(random_state = 0)
        model.fit(X, Y)
        
    else:
        model = sklearn.linear_model.LinearRegression()
        model.fit(X, Y)
    
    predicted = model.predict(XForecast)
    
    return predicted
    
@xw.func
@xw.arg("X", np.array, ndim = 2)
@xw.arg("Y", np.array, ndim=2)
@xw.arg("XForecast", np.array, ndim = 2)
def nn_simple(X, Y, XForecast, hiddenLayerNodes, classification, activation, epochs, batchSize, lr):

    X = X.astype(np.float)
    XForecast = XForecast.astype(np.float)
    Y = Y.astype(np.float)

    X, XForecast = scale(X, XForecast)

    if classification:
        metric = 'accuracy'
        if Y.shape[1]>1:
            # ASSUMES labels are one hot encoded
            output_activation = "softmax"
            loss = "categorical_crossentropy"
        else:
            # ASSUMES labels of 0 or 1
            output_activation = "sigmoid"
            loss = 'binary_crossentropy'
    else:
        output_activation = "linear"
        metric = "mae"
        loss = "mse"
        
    model = Sequential()
    initializer = initializers.HeUniform()
    
    # First layer added separately due to input_dim argument
    model.add(Dense(hiddenLayerNodes[0], input_dim = X.shape[1], kernel_initializer = initializer, activation = activation))
    
    # Only loop to add middle layers except first
    for i in range(1, len(hiddenLayerNodes)):
        model.add(Dense(hiddenLayerNodes[i], activation = activation, kernel_initializer = initializer))
        
    # Add last layer separately due to output activation function
    model.add(Dense(Y.shape[1], activation = output_activation, kernel_initializer = initializer))

    opt = keras.optimizers.Adam(learning_rate = float(lr))
    model.compile(loss=loss, optimizer = opt, metrics = metric)
    history = model.fit(X, Y, epochs = int(epochs), batch_size = int(batchSize), verbose = 1)


    dirname = os.path.dirname(__file__)
    
    f = open(os.path.join(dirname, "nerual net model log.txt"), "w")
    f.write(str(history.history))
    f.write('\n')
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.close()

    predicted = model.predict(XForecast)
    return predicted

#Se till att alla listor initieras korrekt, RNN- done
#Lägg till RNN/GRU/LSTM i valbara menyn- done
#Börja skriva RNN-koden, utgå från NN

@xw.func
@xw.arg("X", np.array, ndim = 2)
@xw.arg("Y", np.array, ndim=2)
@xw.arg("XForecast", np.array, ndim = 2)
def svm_py(X, Y,XForecast, kernel, degree, C, gamma, classification):

    X = X.astype(np.float)
    XForecast = XForecast.astype(np.float)
    Y = Y.astype(np.float)

    X, XForecast = scale(X, XForecast)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)
    
    if classification:
        model = sklearn.svm.SVC(C=float(C), kernel=kernel, degree=int(degree), gamma=float(gamma))
    else:
        model = sklearn.svm.SVR(C=float(C), kernel=kernel, degree=int(degree), gamma=float(gamma))

    model.fit(x_train, y_train)

    dirname = os.path.dirname(__file__)
    
    f = open(os.path.join(dirname, "SVM model log.txt"), "w")
    f.write("Artificial R^2 with train/test split")
    f.write('\n')
    f.write(str(model.score(x_test, y_test)))
    f.close()

    model.fit(X, Y)
    predicted = model.predict(XForecast)
    return predicted
    

if __name__ == "__main__":
   xw.Book("myproject.xlsm").set_mock_caller()
   main()
