import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load model & vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Flipkart Review Sentiment", layout="centered")

st.title("🛒 Flipkart Product Review Sentiment Analysis")
st.write("Enter a product review and get its sentiment instantly.")

review = st.text_area("✍️ Enter Review Text")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned_review = clean_text(review)
        vector = vectorizer.transform([cleaned_review])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.success("Positive Review ")
        else:
            st.error(" Negative Review ")
