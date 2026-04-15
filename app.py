import streamlit as st
import pickle
from lime.lime_text import LimeTextExplainer


# Page settings (important for professional look)
st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🛡️",
    layout="centered"
)


# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

explainer = LimeTextExplainer(class_names=["Non-toxic", "Toxic"])


# Header
st.title("🛡️ Toxic Comment Detector")
st.caption("Explainable AI-based moderation system")


# Input box
user_input = st.text_area(
    "Enter a comment",
    placeholder="Type a comment here..."
)


# Prediction function
def predict_proba(texts):
    vectors = vectorizer.transform(texts)
    return model.predict_proba(vectors)


# Button centered
col1, col2, col3 = st.columns([1,2,1])

with col2:
    predict_button = st.button("Analyze Comment")


# Prediction output
if predict_button and user_input.strip() != "":

    prediction = model.predict(vectorizer.transform([user_input]))

    st.divider()

    if prediction[0] == 1:
        st.error("⚠️ Toxic comment detected")
    else:
        st.success("✅ Non-toxic comment")

    st.subheader("Explanation (word influence)")

    explanation = explainer.explain_instance(user_input, predict_proba)

    for word, weight in explanation.as_list():
        st.write(f"**{word}** : {round(weight, 3)}")


# Footer
st.divider()
st.caption("Built using NLP + Explainable AI (LIME)")