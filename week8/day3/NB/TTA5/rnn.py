import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM, GRU, Embedding


# Sample data: Sequences of numbers
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
y = np.array([4, 7, 10, 13])  # The next number in the sequence


# Reshaping data for RNN
X = X.reshape((X.shape[0], X.shape[1], 1))
print(X)

# Building the RNN Model
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(3, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=200, verbose=0)

# Test the model
test_input = np.array([8, 9, 10])
test_input = test_input.reshape((1, 3, 1))
predicted = model.predict(test_input, verbose=0)
print(f"Predicted value: {int(predicted)}")



# Sample text data
sentences = [
    "I am from Brazil, so at school I had to learn how to speak Portuguese",
    "She is from Japan, so at school she had to learn how to speak Japanese",
    "He is from Spain, so at school he had to learn how to speak Spanish",
    "As a Canadian, in my school, French was a compulsory language to learn",
    "Living in India exposes you to many languages, but Hindi is often a necessity",
    "In the USA, many students choose to learn Spanish as a second language",
    "In Russia, understanding Russian is essential, especially in the education system",
    "In China, Mandarin is the primary language taught in schools",
    "Italy is known for its beautiful language, so Italian is key in their schools",
    "In Germany, learning German is a fundamental part of their education system",
    "Being in Egypt, it is common to learn Arabic in schools",
    # Add more sentences following this pattern
]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1

# Create input sequences and labels
input_sequences = []
for line in sentences:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Predictors and label
X, labels = input_sequences[:,:-1],input_sequences[:,-1]
y = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Define the model: try with LSTM and check the results
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

#---------------- Define the model: try with GRU and check the results
# model = Sequential()
# model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
# model.add(LSTM(150))
# model.add(Dense(total_words, activation='softmax'))


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit the model
model.fit(X, y, epochs=100, verbose=1)

#predict
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len-1, padding='pre')
    prediction = model.predict(sequence, verbose=0)
    index = np.argmax(prediction)
    return tokenizer.index_word[index]

#test
input_text = "I am from Brazil, so at school I had to learn how to speak..."
predicted_word = predict_next_word(input_text)
print("Input:", input_text, "\nPredicted next word:", predicted_word)

# In this example:

# - We create a small dataset of sentences that follow a similar structure but vary in the subject country and language.
# - The model is a simple LSTM network, suitable for this kind of sequence prediction task.
# - We use Keras' `Tokenizer` to convert text to sequences of integers, which are then used to train the model.
# - The model is trained to predict the next word in a sequence, given the preceding words.
# - After training, you can test the model with partial sentences to see if it predicts the correct next word.

# This example illustrates how a sequence model can learn from context that extends beyond immediate adjacent words, though its effectiveness will significantly depend on the training data's size and diversity. In real-world applications, much larger datasets and more complex models would be required for high accuracy.
