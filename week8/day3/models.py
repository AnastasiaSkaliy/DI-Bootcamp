import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Sample data: Sequences of numbers
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
y = np.array([4, 7, 10, 13])  # The next number in the sequence

X = X.reshape((X.shape[0], X.shape[1], 1))

# Building the RNN Model
model = Sequential()
model.add(SimpleRNN(5, activation='relu', input_shape=(3, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=10, verbose=0) 

# Test the model
test_input = np.array([8, 9, 10])
test_input = test_input.reshape((1, 3, 1))
predicted = model.predict(test_input, verbose=0)
print(f"Predicted value: {predicted[0][0]}")

#output
#Predicted value: [[11.153959]]
