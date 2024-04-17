import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

if "trained_model" not in st.session_state:
    st.session_state.model = None

if "predictors" not in st.session_state:
    st.session_state.predictors = None

if "label" not in st.session_state:
    st.session_state.label = None

if "max_sequence_len" not in st.session_state:
    st.session_state.max_sequence_len = None

if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
        
def app():
    st.title('Text Embedding')
    st.write('Text Embedding is a technique to convert text data into numerical \
             data. It is used to convert text data into numerical data so that it \
             can be used in machine learning models. There are many techniques to \
             convert text data into numerical data. Some of the popular techniques \
             are Bag of Words, TF-IDF, Word2Vec, GloVe, etc.')

    # Example text data
    text = st.text_area('Enter text data:')

    if st.sidebar.button('Prepare Data'):    
        # Tokenize the text (convert words to integers)
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts([text])
        st.session_state.tokenizer = tokenizer

        vocab_size = len(tokenizer.word_index) + 1
        sequences = tokenizer.texts_to_sequences([text])[0]

        # Create input sequences and labels
        input_sequences = []
        for i in range(1, len(sequences)):
            n_gram_sequence = sequences[:i+1]
            input_sequences.append(n_gram_sequence)

        # Pad sequences to ensure uniform input size
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        st.session_state.max_sequence_len = max_sequence_len

        # Create predictors and label
        predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
        st.session_state.predictors = predictors
        st.session_state.label = label    

        # Build the RNN model
        model = Sequential()
        model.add(Embedding(vocab_size, 10, input_length=max_sequence_len-1))
        model.add(SimpleRNN(50, return_sequences=True))
        model.add(SimpleRNN(50))
        model.add(Dense(vocab_size, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        st.session_state.trained_model = model
        st.write('Data is prepared successfully!')

    if st.sidebar.button('Begin Training'):
        model = st.session_state.trained_model
        predictors = st.session_state.predictors
        label = st.session_state.label

        # Train the model
        model.fit(predictors, label, epochs=100, verbose=1)

    # Generate text using the trained model
    text_input = st.text_input('Enter seed text:')

    if st.button('Test the Model'):
        max_sequence_len = st.session_state.max_sequence_len
        generated_text = generate_text(text_input, 10, max_sequence_len)
        st.write(generated_text)


# Function to generate text using the trained model
def generate_text(seed_text, next_words, max_sequence_len):
    tokenizer = st.session_state.tokenizer
    model = st.session_state.trained_model

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

if __name__=='__main__':
    app()   
