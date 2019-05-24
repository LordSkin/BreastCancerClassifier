import network

data = network.read_data()
train_data, train_labels, test_data, test_labels = network.separate_sets(data)

model = network.create_model()

print("uczenie:")
model.fit(x=train_data, y=train_labels, epochs=20, batch_size=32)

print("testowanie:")
evaluate = model.evaluate(x=test_data, y=test_labels)
print("wynik:")
print(evaluate[1])
