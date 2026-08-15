import streamlit as st
import joblib

# -----------------------------
# Load model and vectorizer
# -----------------------------
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "spam_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------
st.title("📧 Spam Email Detector")

st.write(
    "Enter an email or message below and our Machine Learning "
    "model will predict whether it is spam or not."
)

st.divider()


# -----------------------------
# Email input
# -----------------------------
message = st.text_area(
    "📨 Enter your email/message:",
    height=220,
    placeholder="Paste your email or message here..."
)


# -----------------------------
# Check button
# -----------------------------
if st.button("🔍 Check Email", use_container_width=True):

    if message.strip() == "":
        st.warning("⚠️ Please enter an email or message first.")

    else:

        # Convert text to TF-IDF
        message_tfidf = vectorizer.transform([message])

        # Prediction
        prediction = model.predict(message_tfidf)[0]

        # Probability
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(message_tfidf)[0]
            confidence = max(probability) * 100
        else:
            confidence = None

        st.divider()

        # Display result
        if prediction == 1:

            st.error("🚨 SPAM EMAIL")

            if confidence is not None:
                st.write(f"**Confidence: {confidence:.2f}%**")

            st.warning(
                "This message contains patterns that are commonly "
                "associated with spam."
            )

        else:

            st.success("✅ NOT SPAM")

            if confidence is not None:
                st.write(f"**Confidence: {confidence:.2f}%**")

            st.info(
                "This message appears to be a normal/legitimate message."
            )
            
            st.divider()

st.subheader("📊 Model Performance")

st.write(
    "The spam detector was evaluated using accuracy, "
    "precision, recall, and F1-score."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "98.48%")

with col2:
    st.metric("Precision", "98.53%")

with col3:
    st.metric("Recall", "89.93%")

with col4:
    st.metric("F1 Score", "94.04%")
    
    st.divider()

st.subheader("📈 Model Comparison")

st.write("Comparison of the three machine learning models tested:")

model_names = [
    "Naive Bayes",
    "Logistic Regression",
    "SVM"
]

model_accuracies = [
    0.960538,
    0.973094,
    0.984753
]

chart_data = {
    "Model": model_names,
    "Accuracy": model_accuracies
}

st.bar_chart(
    chart_data,
    x="Model",
    y="Accuracy"
)