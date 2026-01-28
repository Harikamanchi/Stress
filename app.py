import streamlit as st
import pickle
import re
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# =========================
# NLTK setup (SAFE)
# =========================
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# =========================
# Load model & tokenizer
# =========================
model = load_model("stress_bilstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# =========================
# Text preprocessing
# (NO LEMMATIZATION → SAFE)
# =========================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Stress Detection using NLP")

st.title("🧠 Stress Detection using NLP")
st.write("Enter a sentence to detect stress level")

user_input = st.text_area("Enter text here")

THRESHOLD = 0.6

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        processed = preprocess_text(user_input)
        seq = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(seq, maxlen=100)

        pred_prob = model.predict(padded)[0][0]

        st.write(f"🔍 Stress Probability: {pred_prob:.2f}")

        if pred_prob >= THRESHOLD:
            st.error("⚠️ Stress Detected")
        else:
            st.success("✅ No Stress Detected")
