import random

from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.models import Sequential

import arguments
from data import get_data

from functools import partial

# command line arguments
args = arguments.args

# values to be selected
lstm1_units = [50, 100]
dropout1 = [0.1, 0.2]
lstm2_units = [100, 200, 300]
dropout2 = list(dropout1)


def attr(all_possibilities):
    return partial(random.randint, 0, len(all_possibilities) - 1)


attrs = [attr(lstm1_units), attr(dropout1), attr(lstm2_units), attr(dropout2)]

# training and test data
x_train, y_train, x_test, y_test = get_data()

# cache for keras output
cache = {}


# minimized function
def optimize(current_attrs):
    values = (lstm1_units[current_attrs[0]],
              dropout1[current_attrs[1]],
              lstm2_units[current_attrs[2]],
              dropout2[current_attrs[3]])
    print("Optimizing function for attrs: " + str(current_attrs) + " (values: " + str(values) + ")")

    if values not in cache:
        model = Sequential()
        model.add(LSTM(units=values[0], return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dropout(values[1]))
        model.add(LSTM(values[2]))
        model.add(Dropout(values[3]))
        model.add(Dense(y_train.shape[1]))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=args.epochs, batch_size=32, validation_split=0.15,
                  callbacks=[EarlyStopping(patience=args.early_stopping_patience)])
        cache[values] = [model.evaluate(x_test, y_test)[1]]

    print("Result: " + str(cache[values]))
    return cache[values]
