import random

import numpy as numpy
from keras import optimizers
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras.models import Sequential


def read_data():
    '''
    Zczytuje dane z pliku wdbc.data i zapisuje do macierzy w 32 kolumnach - pierwsza kolumna jest pomijana
    :return: macierz Ax32 gdzie A to liczba rekordów
    '''
    load_data = numpy.loadtxt(dtype=float, usecols=(range(1, 32)), fname="wdbc.data",
                              delimiter=',')
    print(load_data)
    return load_data


def separate_sets(data, test_set=0.3):
    '''
    Rozdziela macierz danych na 4 zbiory - dane treningowe, etykiety treningowe, dane testowe, etykiety testowe
    :param data: macierz z danymi
    :param test_set: stosunek liczności zbioru testowego do wszystkich danych - domyslnie 0.3 (czyli w zbiorze treningowym jest 0.7 danych)
    :return: training_data, training_labels, test_data, test_labels
    '''
    sets_border = int(random.randint(0, int(data.shape[0] * (1.0 - test_set))))
    test_set_size = int(data.shape[0] * test_set)
    test_data = data[sets_border:sets_border + test_set_size, :]
    train_data = numpy.append(data[0:sets_border, :], data[sets_border + test_set_size:data.shape[0] - 1, :], axis=0)

    return train_data[:, 1:], train_data[:, 0], test_data[:, 1:], test_data[:, 0]


def filter_only_selected_attributes(data, attributes):
    '''
    Wybiera tylko wyznaczone kolumny z macierzy data
    :param data: macierz z danymi
    :param attributes: lista interesujących nas numerów kolumn
    :return: filtered_data
    '''
    return data[:, attributes]


def create_model(hidden_layer_neurons=16, dropout=0.2, learning_rate=0.4, momentum=0.09, learning_rate_decay=0.01,
                 attributes=30):
    '''
    Tworzy model sieci neuronowej według podanych parametrów
    :param hidden_layer_neurons: liczba neuronów w warstwie ukrytej
    :param dropout: współczynnik dropoutu
    :param learning_rate: współczynnik uczenia
    :param momentum: wartośc momentum
    :param learning_rate_decay: spadek współczynnika uczenia
    :param attributes: liczba atrybutów w danych treningowych
    :return: Model sieci
    '''
    model = Sequential()

    # warstwa normlizacji
    model.add(Dense(units=attributes, input_dim=attributes, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))

    # warstwa ukryta
    model.add(Dense(units=hidden_layer_neurons, activation='relu', input_dim=attributes))
    model.add(Dropout(dropout))

    # warstwa wyjsciowa
    model.add(Dense(units=1, activation='sigmoid'))

    sgd = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=learning_rate_decay, nesterov=False)

    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model
