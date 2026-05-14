import streamlit as st
import joblib
import re
import string
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load model
model = joblib.load('mental_health_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Clean text
def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'\d+', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return ' '.join(words)

# Emotion keywords
emotion_keywords = {
    'Stress': ['stress', 'pressure', 'overwhelmed'],
    'Anxiety': ['anxious', 'nervous', 'panic', 'fear'],
    'Depression': ['sad', 'hopeless', 'depressed', 'lonely'],
    'Happiness': ['happy', 'joy', 'excited', 'good'],
    'Anger': ['angry', 'mad', 'furious', 'annoyed']
}

# Emotion score function
def calculate_emotion_scores(text):

    text = text.lower()

    scores = {}

    for emotion, keywords in emotion_keywords.items():

        score = sum(word in text for word in keywords)

        scores[emotion] = score

    return scores

# Page config
st.set_page_config(
    page_title="Mental Health Detection",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stApp {
    background: linear-gradient(to right, #141E30, #243B55);
    color: white;
}

h1, h2, h3 {
    color: #00FFD1;
}

textarea {
    background-color: #1E1E1E !important;
    color: white !important;
}

.stButton>button {
    background-color: #00FFD1;
    color: black;
    border-radius: 10px;
    height: 50px;
    width: 200px;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #00C9A7;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🧠 Dashboard")
st.sidebar.write("Mental Health Analysis System")

# Title
st.title("🧠 Mental Health Detection from Text")

st.write(
    "Analyze emotions and mental health conditions using Machine Learning"
)

# Input
user_input = st.text_area(
    "✍ Enter your thoughts or feelings:",
    height=200
)

# Analyze button
if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        cleaned_text = clean_text(user_input)

        vectorized_text = vectorizer.transform([cleaned_text])

        prediction = model.predict(vectorized_text)[0]

        probabilities = model.predict_proba(vectorized_text)[0]

        classes = model.classes_

        # Result card
        st.subheader("🧾 Prediction Result")

        st.success(
            f"Predicted Condition: {prediction}"
        )

        # Analytics
        st.subheader("📊 Prediction Analytics")

        prob_df = pd.DataFrame({
            'Condition': classes,
            'Probability': probabilities
        })

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(
            prob_df['Condition'],
            prob_df['Probability']
        )

        ax.set_title("Prediction Confidence")

        st.pyplot(fig)

        # Emotion scores
        st.subheader("😊 Emotion Visualization")

        emotion_scores = calculate_emotion_scores(user_input)

        emotion_df = pd.DataFrame({
            'Emotion': list(emotion_scores.keys()),
            'Score': list(emotion_scores.values())
        })

        st.dataframe(emotion_df)

        # Pie chart
        fig2, ax2 = plt.subplots(figsize=(6, 6))

        ax2.pie(
            emotion_df['Score'],
            labels=emotion_df['Emotion'],
            autopct='%1.1f%%'
        )

        ax2.set_title("Emotion Distribution")

        st.pyplot(fig2)

        # Progress bars
        st.subheader("📈 Emotion Intensity")

        for emotion, score in emotion_scores.items():

            st.write(emotion)

            st.progress(min(score * 25, 100))

# Footer
st.markdown("---")

st.markdown(
    "<center>Made with ❤️ using Streamlit</center>",
    unsafe_allow_html=True
)