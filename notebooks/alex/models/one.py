
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import layers
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping

EPOCHS = 2
NEURONS = 50
TRAIN_YEAR = 2010
VALID_YEAR = 2015
TEST_YEAR = 2016
FINAL_YEAR = 2020
THRESHOLD = 0.05

def prepData ( symbol='EUR_USD', start_year=2010, final_year=2015, threshold=0.05 ):

    # load target
    y = pd.read_csv(f'../../../db/data/merge/secondary/logs_.csv', index_col=0)
    y.index = pd.to_datetime(y.index)
    y = y.replace([np.inf, -np.inf, np.nan], 0)
    y = y[symbol].loc[str(start_year)+'-01-01':str(final_year)+'-12-31']

    # transform target to a classification (1, 0, -1)
    def condition(x, threshold):
        if x > threshold:
            return 1
        elif x < -threshold:
            return -1
        else:
            return 0
    
    y = y.map(lambda x: condition(x, THRESHOLD)).to_numpy()

    encoder = OneHotEncoder(sparse = False)
    y = encoder.fit_transform(y.reshape(-1,1))

    # load features
    X = pd.read_csv(f'../../../db/data/merge/tendency/tendency.csv', index_col=0)
    X.index = pd.to_datetime(X.index)
    X = X.loc[str(start_year)+'-01-01':str(final_year)+'-12-31']
    X = X.filter(regex=f'{symbol[:3]}|{symbol[4:]}|sin|cos')
    X = X.replace([np.inf, -np.inf, np.nan], 0)

    # scaling features
    def scaleData ( data ):
        series = Series(data)
        # prepare data for normalization
        values = series.values
        values = values.reshape((len(values), 1))
        # train the normalization
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(values)
        #print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
        # normalize the dataset and print
        normalized = scaler.transform(values)
        #print(normalized)
        # inverse transform and print
        inversed = scaler.inverse_transform(normalized)
        #print(inversed)

        return normalized, inversed

    for col in X.columns:
        X[col], _ = scaleData(X[col])

    # make windows TODO

    X = X.to_numpy()

    # shift 1 X and y
    y = y[:-1]
    X = X[1:]

    return y, X

y_train, X_train = prepData('EUR_USD', TRAIN_YEAR, VALID_YEAR, THRESHOLD)
y_val, X_val = prepData('EUR_USD', VALID_YEAR, TEST_YEAR-1, THRESHOLD)
y_test, X_test = prepData('EUR_USD', TEST_YEAR, FINAL_YEAR, THRESHOLD)

def prepModel(X , y, neurons=NEURONS):

    model = Sequential()
    model.add(LSTM(neurons, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(neurons , activation='relu'))
    model.add(Dense(y.shape[-1], activation='softmax',))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

model = prepModel(X_train , y_train)

def trainModel( X , y, epochs=EPOCHS):

    early_stopping = EarlyStopping(monitor='accuracy', patience=24, mode='min')

    history = model.fit(X , y, epochs=epochs, callbacks=[early_stopping])

    model.save('one.h5')

    return history 

history = trainModel(X_train, y_train)

results = model.evaluate(X_test, y_test, batch_size=128)

print("test loss, test acc:", results)