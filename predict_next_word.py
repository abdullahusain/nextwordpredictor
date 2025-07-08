# predict_next_word.py
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle

# Load model and tokenizer
model = load_model("next_word_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_seq_len = model.input_shape[1]

def predict_next_word(input_text):
    token_list = tokenizer.texts_to_sequences([input_text.lower()])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=-1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return ""
