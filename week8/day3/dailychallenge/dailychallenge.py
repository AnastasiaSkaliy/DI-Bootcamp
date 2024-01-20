import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the text corpus from the URL
url = 'https://www.gutenberg.org/cache/epub/11/pg11.txt'
response = requests.get(url)
alice_text = response.text

# Preprocess the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([alice_text])

# Convert text to sequences and pad sequences for uniform length
sequences = tokenizer.texts_to_sequences([alice_text])
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

# Print a part of the processed sequence for verification
print("Original Text:")
print(alice_text[:200])

print("\nProcessed Sequences:")
print(padded_sequences[0][:20])

import re
# Step 2: Create a preprocess function using regular expressions
def preprocess_text(text):
    # Remove non-alphanumeric characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Step 3: Split the text using '*** START' and '*** END', remove irrelevant parts
start_idx = alice_text.find('*** START')
end_idx = alice_text.find('*** END')
corpus = alice_text[start_idx:end_idx]

# Step 4: Print the first 200 characters of the corpus
print("First 200 characters of the corpus:")
print(corpus[:200])

# Step 5: Tokenize the text using Tokenizer() and create vocabulary
tokenizer = Tokenizer()
corpus = preprocess_text(corpus)
tokenizer.fit_on_texts([corpus])

# Calculate total_words
total_words = len(tokenizer.word_index) + 1
print("Total words in the vocabulary:", total_words)

#Prepare Input And Output Data:

# Step 6: Create n-gram sequences
input_sequence = []
for word in tokenizer.texts_to_sequences([corpus])[0]:
    for i in range(1, len(word)):
        n_gram_sequence = word[:i+1]
        input_sequence.append(n_gram_sequence)

# Step 7: Pad the sequences
max_sequence_length = max([len(seq) for seq in input_sequence])
padded_sequences = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='pre')

# Print the first few sequences for verification
print("First few sequences:")
for i in range(5):
    print(padded_sequences[i])

# Print the shape of the padded array
print("Shape of padded array:", padded_sequences.shape)

#Build The Neural Network Model:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Step 8: Build the Neural Network Model
model = Sequential()

# Add an Embedding layer for text representation
embedding_dim = 100
model.add(Embedding(input_dim=total_words, output_dim=embedding_dim, input_length=max_sequence_length))

# Add LSTM layers for processing the sequences
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(100))
model.add(Dropout(0.2))

# Add a Dense layer for output prediction
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

#Compile And Train The Model:

from tensorflow.keras.callbacks import EarlyStopping

# Step 9: Compile the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 10: Train the Model with EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

# Assuming you have labels for the next word in your n-gram sequences
labels = padded_sequences[:, -1]
input_sequences = padded_sequences[:, :-1]

# Train the model
model.fit(input_sequences, labels, epochs=50, batch_size=64, callbacks=[early_stopping])

# Save the model
model.save('text_generation_model.h5')


#Evaluate The Model’s Performance On Test Data:
def generate_text(seed_text, model, tokenizer, max_sequence_length, num_words):
    output_text = seed_text
    
    for _ in range(num_words):
        # Preprocess the seed text
        seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
        seed_padded = pad_sequences([seed_sequence], maxlen=max_sequence_length-1, padding='pre')
        
        # Predict the next word
        predicted_word_index = model.predict_classes(seed_padded, verbose=0)
        
        # Convert the index to the actual word
        predicted_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                predicted_word = word
                break
        
        # Update the seed text for the next iteration
        seed_text += " " + predicted_word
        output_text += " " + predicted_word
    
    return output_text

# Example usage
seed_text = "Alice"
generated_text = generate_text(seed_text, model, tokenizer, max_sequence_length, num_words=50)
print(generated_text)
