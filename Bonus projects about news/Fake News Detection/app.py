import streamlit as st
import pickle
import numpy as np

# 🔹 MUST be the first Streamlit command
st.set_page_config(page_title="Fake News Detector", layout="centered")

# 🔹 Load GloVe embeddings
@st.cache_resource
def load_glove_embeddings(path):
    embeddings = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# 🔹 Text to GloVe vector
def text_to_glove_vector(text, embeddings, dim=100):
    words = text.lower().split()
    vectors = [embeddings[word] for word in words if word in embeddings]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

# 🔹 Load model
@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

# 🔹 Load everything
glove_path = "Data/glove.6B.100d.txt"  # change if needed
model_path = "glove_logistic_model.pkl"

glove_embeddings = load_glove_embeddings(glove_path)
model = load_model(model_path)

# 🔹 Streamlit UI
st.title("📰 Fake News Detection App (GloVe Model)")
st.write("Enter a news article below to check if it's real or fake.")

user_input = st.text_area("News Article or Headline", height=200)

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vec = text_to_glove_vector(user_input, glove_embeddings)
        prediction = model.predict([input_vec])[0]
        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.success(f"Prediction: **{label}**")
