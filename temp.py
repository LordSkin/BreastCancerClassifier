import numpy

import network


data = network.read_data()
train_data, train_labels, test_data, test_labels = network.separate_sets(data)

model = network.create_model()

print("uczenie:")
model.fit(x=train_data, y=train_labels, epochs=20, batch_size=32)

print("testowanie:")
evaluate = model.evaluate(x=test_data, y=test_labels)
print("wynik:")
print(evaluate)

result = numpy.zeros(shape=[50, 3])
for x in range(49):
    data = network.read_data()
    train_data, train_labels, test_data, test_labels = network.separate_sets(data, test_set=(x+1)*0.02)
    model = network.create_model()
    model.fit(x=train_data, y=train_labels, epochs=10, batch_size=32)
    evaluate = model.evaluate(x=test_data, y=test_labels)
    result[x, :] = [x, evaluate[0], evaluate[1]]

print(result)
