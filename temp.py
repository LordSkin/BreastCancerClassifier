import numpy

import network

print("pobieranie danych z pliku:")
data = network.read_data()
print("podział danych na 4 wymagane zbiory:")
train_data, train_labels, test_data, test_labels = network.separate_sets(data)

print("przefiltrowanie tylko interesujących nas cech:")
selected_attributes = [2, 4]
train_data = network.filter_only_selected_attributes(train_data, selected_attributes)
test_data = network.filter_only_selected_attributes(test_data, selected_attributes)

print("stworzenie modelu z domyslnymi parametrami:")
model = network.create_model(attributes=len(selected_attributes))

print("uczenie:")
model.fit(x=train_data, y=train_labels, epochs=20, batch_size=32)

print("testowanie:")
evaluate = model.evaluate(x=test_data, y=test_labels)
print("wynik:")
print(evaluate[1])

# Przykład porównywania sieci dla różnej liczby epok uczenia
result = numpy.zeros(shape=[30, 2])
for x in range(30):
    model = network.create_model(attributes=len(selected_attributes))
    model.fit(x=train_data, y=train_labels, epochs=x, batch_size=32)
    evaluate = model.evaluate(x=test_data, y=test_labels)
    result[x, :] = [x, evaluate[1]]

print(result)
