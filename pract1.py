import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for x in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(x)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

st.title("NLP Pipeline Analyzer")

text = st.text_area(
    "Enter Text",
    "Apple was founded by Steve Jobs. NLP is a branch of AI."
)

if st.button("Analyze"):

    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    filtered = [
        w for w in words
        if w.isalpha() and w.lower() not in stop_words
    ]

    lemmas = [
        lemmatizer.lemmatize(w)
        for w in filtered
    ]

    st.subheader("Tokens")
    st.write(words)

    st.subheader("Sentences")
    st.write(sentences)

    st.subheader("Stopword Removed")
    st.write(filtered)

    st.subheader("Lemmatization")
    st.write(dict(zip(filtered, lemmas)))