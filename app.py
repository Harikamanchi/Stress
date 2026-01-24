import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

model = joblib.load("stress_model.pkl")
tfidf = joblib.load("tfidf.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

st.title("🧠 Stress Detection (NLP)")
user_text = st.text_area("Enter text")

if st.button("Predict"):
    if user_text.strip():
        clean = preprocess(user_text)
        vec = tfidf.transform([clean])
        pred = model.predict(vec)[0]
        st.success("⚠️ Stress Detected" if pred==1 else "✅ No Stress")
    else:
        st.warning("Please enter some text")

