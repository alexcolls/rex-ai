
import os.path
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
from keras import models
import matplotlib.pyplot as plt


EPOCHS = 100
NEURONS = 100
THRESHOLD = 0.05
TRAIN_YEAR = 2010
VALID_YEAR = 2015
TEST_YEAR = 2020
FINAL_YEAR = 2020
DB_PATH = '../../../../db/data/'
SYMBOLS = []

def prepData ( symbol, start_year=2010, final_year=2015, threshold=THRESHOLD, lookback=120 ):

    # load target
    y = pd.read_csv(DB_PATH+'merge/secondary/logs_.csv', index_col=0)

    global SYMBOLS
    SYMBOLS = y.columns

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
    
    y = y.map(lambda x: condition(x, threshold)).to_numpy()

    encoder = OneHotEncoder(sparse = False)
    y = encoder.fit_transform(y.reshape(-1,1))


    # load features
    X = pd.read_csv(DB_PATH+'merge/tendency/tendency.csv', index_col=0)
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

    # make windows
    def makeWindows ( data ):
        pass

    X = X.to_numpy()

    # shift 1 X and y
    y = y[:-1]
    X = X[1:]

    return y, X


def prepModel(X , y, neurons=NEURONS):

    model = Sequential()
    model.add(LSTM(neurons, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(neurons , activation='relu'))
    model.add(Dense(y.shape[-1], activation='softmax',))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def plotHistory ( history ):

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def trainModel ( X , y, symbol, epochs=EPOCHS):

    early_stopping = EarlyStopping(monitor='accuracy', patience=24, mode='min')

    history = model.fit(X , y, epochs=epochs, callbacks=[early_stopping])

    model.save(__file__[:-3]+'_'+symbol+'.h5')

    plotHistory(history)

    return history 



# main for function call.
if __name__ == "__main__":

    prepData('EUR_USD')

    for sym in SYMBOLS:

        print('\n',sym,'\n')

        params = os.path.exists(__file__[:-3]+'_'+sym+'.h5')

        # train model or load model
        model = None
        if not params:

            y_train, X_train = prepData(sym, TRAIN_YEAR, VALID_YEAR)
            y_test, X_test = prepData(sym, VALID_YEAR, TEST_YEAR)

            model = prepModel(X_train , y_train)

            # fit model
            history = trainModel(X_train, y_train, sym)

            print('\n')
            # test model
            results = model.evaluate(X_test, y_test) # batch_size=128)

            print('\n')
            print('test loss:', round(results[0],2), 'test accuracy:', round(results[1],2))
            print('\n')

        else:

            y_val, X_val = prepData(sym, VALID_YEAR, TEST_YEAR-1)
            y_test, X_test = prepData(sym, TEST_YEAR, FINAL_YEAR)

            model = models.load_model(__file__[:-3]+'_'+sym+'.h5')

            results = model.evaluate(X_val, y_val)

            print(results)

            print('\n')
            print('test loss:', round(results[0],2), 'test accuracy:', round(results[1],2))
            print('\n')






