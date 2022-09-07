

import os.path
import numpy as np
import pandas as pd
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import csv

# LSTM model parameters
LAYERS = 7
NEURONS = 100 # updated automatically after knowing X features
LOOKBACK = 120
EPOCHS = 100
THRESHOLD = 0.05 # volatility % bellow = 0
TRAIN_YEAR = 2018
VALID_YEAR = 2021
TEST_YEAR = 2022
FINAL_YEAR = 2022
DB_PATH = '../../db/data/'
SYMBOLS = []


# prepare X and y tensors
def prepData( symbol='EUR_USD', start_year=2010, final_year=2015, threshold=THRESHOLD, lookback=LOOKBACK, load_SYMBOLS=False ):
   
   ### TARGETS ###

    # load history log returns
    y = pd.read_csv(DB_PATH+'merge/secondary/logs_.csv', index_col=0)

    # get symbols list
    if load_SYMBOLS:
        global SYMBOLS
        SYMBOLS = y.columns
        return 0
    
    # cut df and get target
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
    
    # transform log returns to classification values
    y = y.map(lambda x: condition(x, threshold)).to_numpy()

    # encode and reshape data
    encoder = OneHotEncoder(sparse = False)
    y = encoder.fit_transform(y.reshape(-1,1))

    ### FEATURES ###

    # load features
    X = pd.read_csv(DB_PATH+'merge/tendency/tendency.csv', index_col=0)
    X.index = pd.to_datetime(X.index)
    X = X.loc[str(start_year)+'-01-01':str(final_year)+'-12-31']
    X = X.filter(regex=f'{symbol[:3]}|{symbol[4:]}|sin|cos')

    # substract infinites and fill nans
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(method='bfill', inplace=True)
    X.fillna(method='ffill', inplace=True)

    # update model's neurons by number of features
    global NEURONS
    NEURONS = len(X.columns)

    # scaling features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # export pipeline as pickle file
    with open('s3_'+symbol+'.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    # make sequences and output tensors
    def makeSequences( X, y, lookback=lookback ):

        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        X_tensor = []
        y_tensor = []
        for index in range(lookback, X.shape[0]):
            try:
                X_tensor.append(X.iloc[index - lookback:index])
                y_tensor.append(y.iloc[index+1])
            except:
                break
        
        X_tensor = np.array(X_tensor[:-1])
        y_tensor = np.array(y_tensor)

        return X_tensor, y_tensor

    return makeSequences( X, y )


# construct a sequential network
def buildModel( X , y, layers=LAYERS, neurons=NEURONS, dropout=0.2 ):

    print('\n> Building the LSTM model')

    # add dropout neurons
    neurons = int( neurons * ( 1 + dropout ) + 1 )

    print('\nwith', neurons, 'neurons')
    print('\nand', layers, 'layers')
    print('\ndropout:', dropout)

    model = Sequential()

    model.add( LSTM(neurons, activation='tanh', return_sequences=False, input_shape=(X.shape[1], X.shape[2])) )

    model.add( Dropout(dropout) )

    for _ in range(layers):
        model.add( Dense(neurons , activation='tanh') )
        model.add( Dropout(dropout) )

    # final prediction
    model.add( Dense(y.shape[-1], activation='softmax',) )

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    print('\n')

    model.summary()

    print('\n')

    return model


# train & validate the network
def trainModel( model, X, y, X_val, y_val, symbol, epochs=EPOCHS, plot=False):

    early_stopping = EarlyStopping(monitor='accuracy', patience=10, mode='min', restore_best_weights=True)

    history = model.fit(X , y, epochs=epochs, batch_size=LOOKBACK, verbose=1, callbacks=[early_stopping], validation_data=(X_val, y_val))

    model.save(__file__[:-3]+'_'+symbol+'.h5')

    if plot: plotHistory(history)

    return history 


# plot learning scores
def plotHistory( history ):

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return 0


# write testing scores
def makeScores( symbol, accuracy, loss, dataset='train' ):

    file = dataset+'_scores.csv'

    field_names = ['symbol', 'accuracy', 'loss']
    inp = {'symbol': symbol, 'accuracy': accuracy, 'loss': loss}

    # append row to scores csv
    with open(file, 'a') as csv_file:
        dict_object = csv.DictWriter(csv_file, fieldnames=field_names) 
        dict_object.writerow(inp)
        
    return True

    
# main for function call.
if __name__ == "__main__":

    prepData(load_SYMBOLS=True)

    for sym in SYMBOLS:

        print('\n', sym)

        params = os.path.exists(__file__[:-3]+'_'+sym+'.h5')
        # train model or load model
        model = None
        if not params:

            print('\n> Loading and preprocessing data...\n')
            # loading and preparing data
            X_train, y_train = prepData(sym, TRAIN_YEAR, VALID_YEAR-1)
            X_valid, y_valid = prepData(sym, VALID_YEAR, VALID_YEAR)

            print(y_train.shape, X_train.shape)

            # build
            model = buildModel(X_train , y_train, LAYERS, NEURONS, 0.1)

            # fit
            history = trainModel(model, X_train, y_train, X_valid, y_valid, sym)

            print('\n')
            # validate
            results = model.evaluate(X_valid, y_valid)

            # write validation scores to csv
            makeScores( sym, round(results[1],2), round(results[0],2), dataset='val' )

            print('\n')
            print('validation loss:', round(results[0],2), 'validation accuracy:', round(results[1],2))
            print('\n')

        else:

            # load validation & testing data
            X_val, y_val = prepData(sym, VALID_YEAR, TEST_YEAR-1)
            X_test, y_test = prepData(sym, TEST_YEAR, FINAL_YEAR)

            # load model's params
            model = load_model(__file__[:-3]+'_'+sym+'.h5')

            # test
            results = model.evaluate(X_val, y_val)

            print(results)

            # write test scores to csv
            makeScores( sym, round(results[1],2), round(results[0],2), dataset='test' )

            print('\n')
            print('test loss:', round(results[0],2), 'test accuracy:', round(results[1],2))
            print('\n')


# The End
