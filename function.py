import random

from keras.layers import LSTM, Dense, Activation, Dropout
from keras.models import Sequential

from data import get_training_and_test_data

from functools import partial

# training data
x, y = get_training_and_test_data()

# cache for keras output
cache = {}

# minimized function
def optimize(params):
    current_params = (params[0] * 10, params[1] * 10)
    print
    print("Optimizing function for parameters: " + str(current_params))

    if current_params not in cache:
        model = Sequential()
        model.add(LSTM(units=current_params[0], return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(current_params[1]))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1]))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        cache[current_params] = [model.fit(x, y, epochs=1, batch_size=32, validation_split=0.15).history.get('val_acc')[-1]]

    print("result: " + str(cache[current_params]))
    return cache[current_params]

# function parameters
attr_1 = partial(random.randint, 1, 10)
attr_2 = partial(random.randint, 1, 10)
attrs = [attr_1, attr_2]

