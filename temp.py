import numpy as numpy

from keras import optimizers
from keras.layers import Dense, BatchNormalization, Activation
from keras.models import Sequential


def read_data():
    load_data = numpy.loadtxt(dtype=float, usecols=(range(1, 32)), fname="wdbc.data",
                              delimiter=',')
    print(load_data)
    return load_data


def separate_sets(data):
    count = data.shape[0]
    train_size = int(0.7 * count)
    train_data = data[0:train_size, :]
    test_data = data[train_size:count-1, :]
    return train_data, test_data



data = read_data()
train_data, test_data = separate_sets(data)

model = Sequential()
model.add(Dense(units=30, input_dim=30, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('tanh'))

model.add(Dense(units=16, activation='relu', input_dim=30))
model.add(Dense(units=1, activation='sigmoid'))

sgd = optimizers.SGD(lr =0.4, momentum=0.09, decay=0.1, nesterov=True)

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.summary()

model.fit(x=train_data[:, 1:], y=train_data[:, 0], epochs=20, batch_size=16)

print("testowanie:")
evaluate = model.evaluate(x=test_data[:, 1:], y=test_data[:, 0])
print("wynik:")
print(evaluate)
