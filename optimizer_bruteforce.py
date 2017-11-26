import time

from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.models import Sequential

import arguments
from data import get_data

start = time.time()

# command line arguments
args = arguments.args

# values to be selected
lstm1_units = [50, 100]
dropout1 = [0.1, 0.2]
lstm2_units = [100, 200, 300]
dropout2 = list(dropout1)

# training and test data
x_train, y_train, x_test, y_test = get_data()

best_test_accuracy = 0
best_values = ()

for i in range(len(lstm1_units)):
    for j in range(len(dropout1)):
        for k in range(len(lstm2_units)):
            for l in range(len(dropout2)):
                model = Sequential()
                model.add(
                    LSTM(units=lstm1_units[i], return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
                model.add(Dropout(dropout1[j]))
                model.add(LSTM(lstm2_units[k]))
                model.add(Dropout(dropout2[l]))
                model.add(Dense(y_train.shape[1]))
                model.add(Activation("softmax"))
                model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
                model.fit(x_train, y_train, epochs=args.epochs, batch_size=32, validation_split=0.15,
                          callbacks=[EarlyStopping(patience=args.early_stopping_patience)])
                test_accuracy = model.evaluate(x_test, y_test)[1]
                print("Result: " + str(test_accuracy))
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    best_values = (lstm1_units[i], dropout1[j], lstm2_units[k], dropout2[l])

print("Best result: " + str(best_test_accuracy) + " for values: " + str(best_values))
end = time.time()
print(end - start)
