import xlwings as xw
import numpy as np
import pandas as pd
from sklearn import *
import sklearn
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import SimpleRNN, LSTM, GRU
from keras.layers import Dropout
from tensorflow.keras import initializers
import keras
import os
import category_encoders as ce
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor as xgbreg
from xgboost import XGBClassifier as xgbcla

def fill_missing(X, x_forecast, Y, missing_data_method):
    
    if missing_data_method == "None":
        return X, x_forecast, Y

    if missing_data_method == "Drop column":
        X.dropna(axis = 1, inplace = True)
        x_forecast.dropna(axis = 1, inplace = True)
        Y.dropna(axis = 1, inplace = True)

    elif missing_data_method == "Drop row":
        X.dropna(axis=0, inplace=True)
        x_forecast.dropna(axis=0, inplace=True)
        Y.dropna(axis=0, inplace=True)
    
    elif missing_data_method == "Average":
        X.fillna(X.mean(), inplace = True)
        x_forecast.fillna(x_forecast.mean(), inplace = True)
        Y.fillna(Y.mean(), inplace = True)
        
    elif missing_data_method == "Median":
        X.fillna(X.median(), inplace = True)
        x_forecast.fillna(x_forecast.median(), inplace = True)
        Y.fillna(Y.median(), inplace = True)
       
    elif missing_data_method == "Feed backward":
        X.fillna(method = "bfill", inplace = True)
        x_forecast.fillna(method="bfill", inplace = True)
        Y.fillna(method="bfill", inplace = True)
        
    elif missing_data_method == "Feed forward":
        X.fillna(method = "ffill", inplace = True)
        x_forecast.fillna(method="ffill", inplace = True)
        Y.fillna(method="ffill", inplace = True)

    elif missing_data_method == "Zeros":
        X.fillna(0, inplace = True)
        x_forecast.fillna(0, inplace = True)
        Y.fillna(0, inplace = True)
        
    elif missing_data_method == "Linear interpolation":
        X.interpolate(method="linear", axis = 0, inplace = True)
        x_forecast.interpolate(method="linear", axis=0, inplace = True)
        Y.interpolate(method="linear", axis=0, inplace = True)
    
    return X, x_forecast, Y
    

def scale(X, x_forecast, scale_indices = []):
    # Some columns might be one-hot encoded, avoid scaling these
        
    if scale_indices == []:
        return X, x_forecast
    
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
   
    scale_x = X.iloc[:, scale_indices]
    scale_x_forecast = x_forecast.iloc[:, scale_indices]
    
    scaler.fit(scale_x)
    X_tmp = scaler.transform(scale_x)
    x_forecast_tmp = scaler.transform(scale_x_forecast)

    X.iloc[:, scale_indices] = X_tmp
    x_forecast.iloc[:, scale_indices] = x_forecast_tmp

    return X, x_forecast
    
def transform_categorical(X, x_forecast, Y, has_categorical):
    # Currently uses one-hot encoder straight through
    
    if not has_categorical:
        return X, x_forecast, Y, [i for i in range(X.shape[1])]
    
    merged_df = pd.concat([X, x_forecast]) #Make sure that all different kind of values in columns are present
    final_df_x = pd.DataFrame()
    final_df_y = pd.DataFrame()
    scale_indices = []
    
    # Only apply one-hot encoding on the columns which are not numerical (int, float etc)
    for col in merged_df:
        column = merged_df.iloc[:, col]
        if column.dtype == 'object':
            one_hot_tmp = get_dummies(column, prefix=str(col))
            final_df_x = final_df_x.join(one_hot_tmp)
        else:
            final_df_x = final_df_x.join(column)
            scale_indices = scale_indices.append(len(final_df.columns)-1)
    
    for col in Y:
        column = Y.iloc[:, col]
        if column.dtype == 'object':
            one_hot_tmp = get_dummies(column, prefix=str(col))
            final_df_y = final_df_y.join(one_hot_tmp)
        else:
            final_df_y = final_df_y.join(column)
            
    X = final_df_x.iloc[:len(X), :]
    x_forecast = final_df_x.iloc[len(X):, :]
    
    return X, x_forecast, final_df_y, scale_indices
        
    
def main():
    wb = xw.Book.caller()
    sheet = wb.sheets[0]
    if sheet["A1"].value == "Hello xlwings!":
        sheet["A1"].value = "Bye xlwings!"
    else:
        sheet["A1"].value = "Hello xlwings!"


@xw.func
@xw.arg("X", pd.DataFrame, ndim = 2)
@xw.arg("Y", pd.DataFrame)
@xw.arg("x_forecast", pd.DataFrame, ndim = 2)
def knn_py(X, x_forecast, Y, n_neighbors, weights, algorithm, leaf_size, p, classification, contains_categorical):
    
    X, x_forecast, Y = fill_missing(X, x_forecast, Y, missing_data_method)
    X, x_forecast, Y, scale_indices = transform_categorical(X, x_forecast, Y)
    X, x_forecast = scale(X, x_forecast, scale_indices)
    
    if classification:
        model = KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights, algorithm = algorithm, leaf_size = leaf_size, p = p)
    else:
        model = KNeighborsRegressor(n_neighbors = n_neighbors, weights = weights, algorithm = algorithm, leaf_size = leaf_size, p = p)
       
    model.fit(X, y)
    predicted = model.predict(x_forecast)
    return predicted
    
@xw.func
@xw.arg("X", pd.DataFrame, ndim = 2)
@xw.arg("Y", pd.DataFrame)
@xw.arg("x_forecast", pd.DataFrame, ndim = 2)
def xgboost_py(X, x_forecast, Y, learning_rate, max_depth, subsample, colsample_bytree, n_estimators, objective, booster,
                min_split_loss, sampling_method, classification, contains_categorical):
        
    X, x_forecast, Y = fill_missing(X, x_forecast, Y, missing_data_method)
    X, x_forecast, Y, scale_indices = transform_categorical(X, x_forecast, Y)
    X, x_forecast = scale(X, x_forecast, scale_indices)
    
    if classification:
        model = xgbcla(learning_rate = learning_rate, max_depth = max_depth, subsample = subsample, colsample_bytree = colsample_bytree,
                    n_estimators = n_estimators, objective = objective, booster = booster,min_split_loss = min_split_loss, sampling_method = sampling_method)
    else:
        model = xgbreg(learning_rate = learning_rate, max_depth = max_depth, subsample = subsample, colsample_bytree = colsample_bytree,
                    n_estimators = n_estimators, objective = objective, booster = booster,min_split_loss = min_split_loss, sampling_method = sampling_method)
    model.fit(X, Y)
    predicted = model.predict(x_forecast)
    return predicted

@xw.func
def hello(name):
    return f"Hello {name}!"

@xw.func
@xw.arg("X", pd.DataFrame, ndim = 2)
@xw.arg("Y", pd.DataFrame)
@xw.arg("x_forecast", pd.DataFrame, ndim = 2)
def linreg(X, Y, x_forecast, missing_data_method, contains_categorical):

    X, x_forecast, Y = fill_missing(X, x_forecast, Y, missing_data_method)
    X, x_forecast, Y, scale_indices = transform_categorical(X, x_forecast, Y, contains_categorical)

    if len(X.shape) == 1:
        X = X.reshape(-1,1)
        x_forecast = x_forecast.reshape(-1,1)
    
    X, x_forecast = scale(X, x_forecast, scale_indices)

    model = sklearn.linear_model.LinearRegression()
    model.fit(X, Y)
    predicted = model.predict(x_forecast)
    
    return predicted


@xw.func
@xw.arg("X", np.array, ndim = 2, dtype = 'object')
@xw.arg("Y", np.array, dtype = 'object')
@xw.arg("x_forecast", np.array, ndim = 2, dtype = 'object')

def logreg(X, Y, x_forecast, penalty, tolerance, c, solver, multi_class, max_iter, missing_data_method, contains_categorical):

    X, x_forecast, Y = fill_missing(X, x_forecast, Y, missing_data_method)
    X, x_forecast, Y, scale_indices = transform_categorical(X, x_forecast, Y)

    if len(X.shape) == 1:
        X = X.reshape(-1,1)
        x_forecast = x_forecast.reshape(-1,1)
   
    
    X, x_forecast = scale(X, x_forecast, scale_indices)
        
    model = sklearn.linear_model.LogisticRegression(random_state = 0)
    model.fit(X, Y)
    predicted = model.predict(x_forecast)
    
    return predicted
    

# VANILLA NEURAL NETWORK
@xw.func
@xw.arg("X", np.array, ndim = 2, dtype = 'object')
@xw.arg("Y", np.array, ndim=2, dtype = 'object')
@xw.arg("x_forecast", np.array, ndim = 2, dtype = 'object')
def nn_simple(X, Y, x_forecast, hidden_layer_nodes, classification, activation, epochs, batch_size, lr,
                missing_data_method, contains_categorical):

    X, x_forecast, Y = fill_missing(X, x_forecast, Y, missing_data_method)
    X, x_forecast, Y, scale_indices = transform_categorical(X, x_forecast, Y)
    X, x_forecast = scale(X, x_forecast, scale_indices)

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
    model.add(Dense(hidden_layer_nodes[0], input_dim = X.shape[1], kernel_initializer = initializer, activation = activation))
    
    # Only loop to add middle layers except first
    for i in range(1, len(hidden_layer_nodes)):
        model.add(Dense(hidden_layer_nodes[i], activation = activation, kernel_initializer = initializer))
        
    # Add last layer separately due to output activation function
    model.add(Dense(Y.shape[1], activation = output_activation, kernel_initializer = initializer))

    opt = keras.optimizers.Adam(learning_rate = float(lr))
    model.compile(loss=loss, optimizer = opt, metrics = metric)
    history = model.fit(X, Y, epochs = int(epochs), batch_size = int(batch_size), verbose = 1)


    dirname = os.path.dirname(__file__)
    
    f = open(os.path.join(dirname, "nerual net model log.txt"), "w")
    f.write(str(history.history))
    f.write('\n')
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.close()

    predicted = model.predict(x_forecast)
    return predicted


#RECURRENT NEURAL NETWORK
@xw.func
@xw.arg("X", np.array, ndim = 2, dtype = 'object')
@xw.arg("Y", np.array, ndim=2, dtype = 'object')
@xw.arg("x_forecast", np.array, ndim = 2, dtype = 'object')
def rnn_py(X, Y, x_forecast, hidden_layer_nodes, classification, activation, epochs, batch_size, lr,
            rnn_activation, rnn_type, dropout, rnn_dropout, rnn_lookback, missing_data_method, contains_categorical):

    X = X.astype(np.float)
    x_forecast = x_forecast.astype(np.float)
    Y = Y.astype(np.float)

    X, x_forecast, Y = fill_missing(X, x_forecast, Y, missing_data_method)
    X, x_forecast, Y, scale_indices = transform_categorical(X, x_forecast, Y)
    X, x_forecast = scale(X, x_forecast, scale_indices)
    rnn_lookback = int(rnn_lookback)

    dropout = float(dropout)
    rnn_dropout = float(rnn_dropout)
    num_features = X.shape[1]
    rows_train = X.shape[0]-rnn_lookback
    rows_forecast = x_forecast.shape[0]-rnn_lookback

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
        
    x_reshaped = np.array([])
    y_reshaped = np.array([])
    x_forecast_reshaped = np.array([])

    # Reshape the array according to look-back period provided by the user
    for i in range(int(rnn_lookback), int(X.shape[0])):
        x_reshaped = np.append(x_reshaped, X[(i-rnn_lookback):i+1, :])
        y_reshaped = np.append(y_reshaped, Y[i, :])

    for i in range(int(rnn_lookback), int(x_forecast.shape[0])):
        x_forecast_reshaped = np.append(x_forecast_reshaped, x_forecast[(i-rnn_lookback):i+1, :])
    
    try:
        return_sequences = [True for i in range(len(hidden_layer_nodes))]
    except:
        hidden_layer_nodes = [hidden_layer_nodes]
        return_sequences = [True]
    
    return_sequences[-1] = False
    x_reshaped = np.reshape(x_reshaped, (rows_train, rnn_lookback+1, num_features))
    x_forecast_reshaped = np.reshape(x_forecast_reshaped, (rows_forecast, rnn_lookback+1, num_features))
        
    model = Sequential()
    initializer = initializers.HeUniform()

    if rnn_type == "RNN":
        #, input_dim = (x_reshaped.shape[0]'
        model.add(SimpleRNN(units = hidden_layer_nodes[0], input_shape =(rnn_lookback+1, num_features), kernel_initializer = initializer, activation = activation,
                    return_sequences = return_sequences[0], dropout = dropout, recurrent_dropout = rnn_dropout))
        
        for i in range(1, len(hidden_layer_nodes)):
            model.add(SimpleRNN(units = hidden_layer_nodes[i],return_sequences = return_sequences[i], activation = activation, kernel_initializer = initializer,
            dropout = dropout, recurrent_dropout = rnn_dropout))

    elif rnn_type == "LSTM":
        #, input_dim = (x_reshaped.shape[0]
        model.add(LSTM(units = hidden_layer_nodes[0], input_shape =(rnn_lookback+1, num_features), kernel_initializer = initializer, activation = activation,
            return_sequences = return_sequences[0], recurrent_activation = rnn_activation, dropout = dropout, recurrent_dropout = rnn_dropout))
        
        for i in range(1, len(hidden_layer_nodes)):
            model.add(LSTM(hidden_layer_nodes[i],return_sequences = return_sequences[i], activation = activation, kernel_initializer = initializer,
             recurrent_activation = rnn_activation, dropout = dropout, recurrent_dropout = rnn_dropout))
    
    model.add(Dense(Y.shape[1], activation = output_activation, kernel_initializer = initializer))

    opt = keras.optimizers.Adam(learning_rate = float(lr))
    model.compile(loss=loss, optimizer = opt, metrics = metric)
    history = model.fit(x_reshaped, y_reshaped, epochs = int(epochs), batch_size = int(batch_size), verbose = 1)

    dirname = os.path.dirname(__file__)
    
    f = open(os.path.join(dirname, "rnn model log.txt"), "w")
    f.write(str(history.history))
    f.write('\n')
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.close()

    predicted = model.predict(x_forecast_reshaped)
    return predicted

# SUPPORT VECTOR MACHINE
@xw.func
@xw.arg("X", np.array, ndim = 2, dtype = 'object')
@xw.arg("Y", np.array, ndim=2, dtype = 'object')
@xw.arg("x_forecast", np.array, ndim = 2, dtype = 'object')
def svm_py(X, Y,x_forecast, kernel, degree, C, gamma, classification, missing_data_method, contains_categorical):

    X = X.astype(np.float)
    x_forecast = x_forecast.astype(np.float)
    Y = Y.astype(np.float)

    X, x_forecast, Y = fill_missing(X, x_forecast, Y, missing_data_method)
    X, x_forecast, Y, scale_indices = transform_categorical(X, x_forecast, Y)
    X, x_forecast = scale(X, x_forecast, scale_indices)

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
    predicted = model.predict(x_forecast)
    return predicted
    

if __name__ == "__main__":
   xw.Book("myproject.xlsm").set_mock_caller()
   main()



'''
#CONVOLUTIONAL NEURAL NETWORK
@xw.func
@xw.arg("X", np.array, ndim = 2)
@xw.arg("Y", np.array, ndim=2)
@xw.arg("x_forecast", np.array, ndim = 2)
def cnn_py(X, Y, x_forecast, outputLayerNodes, classification, activation, epochs, batch_size, lr,
            dimension, filters, kernelSize, groups, strides, padding, dilationRate):

    X = X.astype(np.float)
    x_forecast = x_forecast.astype(np.float)
    Y = Y.astype(np.float)

    X, x_forecast = scale(X, x_forecast)
    rnn_lookback = int(rnn_lookback)

    num_features = X.shape[1]
    rows_train = X.shape[0]-rnn_lookback
    rows_forecast = x_forecast.shape[0]-rnn_lookback

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
    inputShape = []

    #Different type of layer used depending on what dimension user put in
    if dimension == 1:
        model.add(Conv1D(filters[0], kernelSize, strides, padding, dilation_rate = dilationRate, groups = groups, activation = activation, input_shape = inputShape)
        model.add(MaxPooling1D(pool_size = 2, filters = filters[0]))
        
        for i in range(1, len(filters)):
            model.add(Conv1D(filters[i], kernelSize, strides, padding, dilation_rate = dilationRate, groups = groups, activation = activation)
            model.add(MaxPooling1D(pool_size = 2, filters = filters[i]))

    elif dimension == 2:
        model.add(Conv2D(filters[0], kernelSize, strides, padding, dilation_rate = dilationRate, groups = groups, activation = activation, input_shape = inputShape)
        model.add(MaxPooling2D(pool_size = 2, filters = filters[0]))
        
        for i in range(1, len(filters)):
            model.add(Conv2D(filters[i], kernelSize, strides, padding, dilation_rate = dilationRate, groups = groups, activation = activation)
            model.add(MaxPooling2D(pool_size = 2, filters = filters[i]))
    
    else:
        model.add(Conv3D(filters[0], kernelSize, strides, padding, dilation_rate = dilationRate, groups = groups, activation = activation, input_shape = inputShape)
        model.add(MaxPooling3D(pool_size = 2, filters = filters[0]))
        
        for i in range(1, len(filters)):
            model.add(Conv3D(filters[i], kernelSize, strides, padding, dilation_rate = dilationRate, groups = groups, activation = activation)
            model.add(MaxPooling3D(pool_size = 2, filters = filters[i]))
        
    model.add(Flatten())
    model.add(Dense(outputLayerNodes, activation = activation, kernel_initializer = initializer)
    model.add(Dense(Y.shape[1], activation = output_activation, kernel_initializer = initializer))

    opt = keras.optimizers.Adam(learning_rate = float(lr))
    model.compile(loss=loss, optimizer = opt, metrics = metric)
    history = model.fit(x_reshaped, y_reshaped, epochs = int(epochs), batch_size = int(batch_size), verbose = 1)

    dirname = os.path.dirname(__file__)
    
    f = open(os.path.join(dirname, "rnn model log.txt"), "w")
    f.write(str(history.history))
    f.write('\n')
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.close()

    predicted = model.predict(x_forecast_reshaped)
    return predicted

'''
