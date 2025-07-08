# app.py
import streamlit as st
from predict_next_word import predict_next_word

st.title("Next Word Predictor (LSTM Autocomplete)")

user_input = st.text_input("Type your sentence here:")

if user_input.strip():
    next_word = predict_next_word(user_input)
    if next_word:
        st.markdown(f"**Predicted Next Word:** `{next_word}`")
        st.markdown(f"**Full Sentence Suggestion:** `{user_input} {next_word}`")
    else:
        st.warning("Couldn't predict a word. Try a longer input.")
