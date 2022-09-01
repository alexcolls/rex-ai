
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


X = pd.read_csv('../../db/data/tertiary/2010/logs_.csv', index_col=0)
X.reset_index(inplace=True, drop=True)
X = X[1:]

y = pd.read_csv('../../db/data/secondary/2010/logs_.csv', index_col=0)
y.reset_index(inplace=True, drop=True)
y = y[:-1]

ln = len(X)
prop = 0.5
Ntrain = int(ln*prop)

X_train = X[:Ntrain]
X_test  = X[Ntrain:]
y_train = y[:Ntrain]
y_test  = y[Ntrain:] 

print(X.shape, y.shape)

####################
# Cutting function #
####################
def stateful_cut(arr, batch_size, T_after_cut):
    if len(arr.shape) != 3:
        # N: Independent sample size,
        # T: Time length,
        # m: Dimension
        print("ERROR: please format arr as a (N, T, m) array.")

    N = arr.shape[0]
    T = arr.shape[1]

    # We need T_after_cut * nb_cuts = T
    nb_cuts = int(T / T_after_cut)
    if nb_cuts * T_after_cut != T:
        print("ERROR: T_after_cut must divide T")

    # We need batch_size * nb_reset = N
    # If nb_reset = 1, we only reset after the whole epoch, so no need to reset
    nb_reset = int(N / batch_size)
    if nb_reset * batch_size != N:
        print("ERROR: batch_size must divide N")

    # Cutting (technical)
    cut1 = np.split(arr, nb_reset, axis=0)
    cut2 = [np.split(x, nb_cuts, axis=1) for x in cut1]
    cut3 = [np.concatenate(x) for x in cut2]
    cut4 = np.concatenate(cut3)
    return(cut4)

##
# Model
##
nb_units = 10

model = Sequential()
model.add(LSTM( units=nb_units, stateful=True))
model.add(TimeDistributed(Dense(activation='relu', units=1)))
model.add(LSTM(units=nb_units, stateful=True))
model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse', 'accuracy', 'r2'])


epochs = 100

# When nb_reset = 1, we do not need to reinitialize states
history = model.fit(X_train, y_train, epochs = epochs, 
                    batch_size = 24, shuffle=False,
                    validation_data=(X_test, y_test))

history2 = model.fit(X_train, y_train, batch_size=(8, 28, 1000))

def plotting(history):
    plt.plot(history.history['loss'], color = "red")
    plt.plot(history.history['val_loss'], color = "blue")
    red_patch = mpatches.Patch(color='red', label='Training')
    blue_patch = mpatches.Patch(color='blue', label='Test')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xlabel('Epochs')
    plt.ylabel('MSE loss')
    plt.show()

plt.figure(figsize=(10,8))
plotting(history) # Evolution of training/test loss

##
# Visual checking for a time series
##
## Mime model which is stateless but containing stateful weights
model_stateless = Sequential()
model_stateless.add(LSTM(input_shape=(None, dim_in),
               return_sequences=True, units=nb_units))
model_stateless.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
model_stateless.compile(loss = 'mse', optimizer = 'rmsprop')
model_stateless.set_weights(model.get_weights())

## Prediction of a new set
i = 0 # time series selected (between 0 and N-1)
x = X_train[i]
y = y_train[i]
y_hat = model_stateless.predict(np.array([x]))[0]

for dim in range(3): # dim = 0 for y1 ; dim = 1 for y2 ; dim = 2 for y3.
    plt.figure(figsize=(10,8))
    plt.plot(range(T), y[:,dim])
    plt.plot(range(T), y_hat[:,dim])
    plt.show()

## Conclusion: works almost perfectly.