import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Read the document
with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()

# 2. Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

# 3. Create input sequences
input_sequences = []

for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i + 1])

# Padding
max_seq_len = max(len(seq) for seq in input_sequences)

input_sequences = np.array(
    pad_sequences(
        input_sequences,
        maxlen=max_seq_len,
        padding="pre"
    )
)

# Separate input and output
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# 4. Build and Train RNN Model
model = Sequential([
    Embedding(total_words, 10, input_length=max_seq_len - 1),
    SimpleRNN(50),
    Dense(total_words, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.fit(
    X,
    y,
    epochs=100,
    verbose=1
)

# 5. Predict Next Word
def predict_next_word(seed_text):

    seed_text = seed_text.lower()

    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    if len(token_list) == 0:
        print("Word not found in the document.")
        return

    token_list = pad_sequences(
        [token_list],
        maxlen=max_seq_len - 1,
        padding="pre"
    )

    predicted = model.predict(token_list, verbose=0)

    predicted_word_index = np.argmax(predicted, axis=-1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            print("Predicted next word:", word)
            return

# 6. Take input from user
user_input = input("Enter a word or phrase: ")

predict_next_word(user_input)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Read the document
with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()

# 2. Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

print("Total words:", total_words)

# 3. Create input sequences
input_sequences = []

for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i + 1])

# Check sequences
if len(input_sequences) == 0:
    print("No input sequences found.")
    exit()

# Padding
max_seq_len = max(len(seq) for seq in input_sequences)

input_sequences = np.array(
    pad_sequences(
        input_sequences,
        maxlen=max_seq_len,
        padding="pre"
    )
)

# Separate input and output
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

print("Input shape:", X.shape)
print("Output shape:", y.shape)

# 4. Build GRU Model
model = Sequential([
    Embedding(
        input_dim=total_words,
        output_dim=10,
        input_length=max_seq_len - 1
    ),

    GRU(50),

    Dense(
        total_words,
        activation="softmax"
    )
])

# 5. Compile the model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# 6. Train the model
model.fit(
    X,
    y,
    epochs=100,
    verbose=1
)

# 7. Predict Next Word
def predict_next_word(seed_text):

    seed_text = seed_text.lower()

    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    if len(token_list) == 0:
        print("Word not found in the document.")
        return

    # Keep only required sequence length
    token_list = token_list[-(max_seq_len - 1):]

    # Padding
    token_list = pad_sequences(
        [token_list],
        maxlen=max_seq_len - 1,
        padding="pre"
    )

    # Prediction
    predicted = model.predict(
        token_list,
        verbose=0
    )

    predicted_word_index = np.argmax(
        predicted,
        axis=-1
    )[0]

    # Find predicted word
    for word, index in tokenizer.word_index.items():

        if index == predicted_word_index:

            print("Predicted next word:", word)

            return

    print("Could not predict the next word.")


# 8. Take input from user
user_input = input("Enter a word or phrase: ")

predict_next_word(user_input)
