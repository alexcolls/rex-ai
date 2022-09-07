
import os.path
import numpy as np
import pandas as pd
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from pandas import Series
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
from keras import models
import matplotlib.pyplot as plt
import pickle


EPOCHS = 100
NEURONS = 200
THRESHOLD = 0.05
TRAIN_YEAR = 2018
VALID_YEAR = 2021
TEST_YEAR = 2022
FINAL_YEAR = 2022
DB_PATH = '../../db/data/'
SYMBOLS = []


def prepData ( symbol, start_year=2010, final_year=2015, threshold=THRESHOLD, lookback=120, load_SYMBOLS=False ):

    ### TARGET ###

    y = pd.read_csv(DB_PATH+'merge/secondary/logs_.csv', index_col=0)

    if load_SYMBOLS:
        global SYMBOLS
        SYMBOLS = y.columns
        return True

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

    ### FEATURES ###

    # load features
    X = pd.read_csv(DB_PATH+'merge/tendency/tendency.csv', index_col=0)
    X.index = pd.to_datetime(X.index)
    X = X.loc[str(start_year)+'-01-01':str(final_year)+'-12-31']
    
    # substract infinites and fill nans
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(method='bfill', inplace=True)
    X.fillna(method='ffill', inplace=True)

    # scaling features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # export pipeline as pickle file
    with open("scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    # make sequences TODO
    def makeSequences ( data, periods=lookback ):
        pass

    X = X.to_numpy()

    # shift 1 X and y
    y = y[:-1]
    X = X[1:]

    return y, X


def buildModel ( X , y, neurons=NEURONS ):

    model = Sequential()
    model.add(LSTM(neurons, activation='tanh', input_shape=(X.shape[1], 1)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(neurons , activation='tanh'))
    model.add(layers.Dropout(0.2))
    model.add(Dense(y.shape[-1], activation='softmax',))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def trainModel ( X , y, symbol, epochs=EPOCHS):

    early_stopping = EarlyStopping(monitor='accuracy', patience=10, mode='min')

    history = model.fit(X , y, epochs=epochs, callbacks=[early_stopping])

    model.save(__file__[:-3]+'_'+symbol+'.h5')

    return history 


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

    return 0


# main for function call.
if __name__ == "__main__":

    prepData('EUR_USD', load_SYMBOLS=True)

    for sym in SYMBOLS:

        print('\n',sym,'\n')

        params = os.path.exists(__file__[:-3]+'_'+sym+'.h5')

        # train model or load model
        model = None
        if not params:

            y_train, X_train = prepData(sym, TRAIN_YEAR, VALID_YEAR-1)
            y_test, X_test = prepData(sym, VALID_YEAR, VALID_YEAR)

            model = buildModel(X_train , y_train)

            # fit model
            history = trainModel(X_train, y_train, sym)

            # plotHistory(history)

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


# END
