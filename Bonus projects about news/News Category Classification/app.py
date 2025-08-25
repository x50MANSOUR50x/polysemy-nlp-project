import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (first time only)
nltk.download("stopwords")

# Load model and vectorizer
with open("news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing function
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

# Mapping of class labels
label_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

# Streamlit UI
st.set_page_config(page_title="News Category Classifier", layout="centered")
st.title("📰 News Category Classifier")
st.markdown("Enter a news article to predict whether it's about **World, Sports, Business**, or **Sci/Tech**.")

user_input = st.text_area("🧾 Paste your news article here:", height=200)

if st.button("🔍 Predict Category"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean_text = preprocess(user_input)
        features = vectorizer.transform([clean_text])
        prediction = model.predict(features)[0]
        category = label_map[prediction]
        st.success(f"📢 This news is about: **{category}**")