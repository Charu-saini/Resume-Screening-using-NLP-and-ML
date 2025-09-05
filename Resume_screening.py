import streamlit as st
import pickle
import PyPDF2
import re

# ---------------------------
# Load Models & Vectorizers
# ---------------------------
model_category = pickle.load(open("model_category.pkl", "rb"))
tfidf_category = pickle.load(open("tfidf_category.pkl", "rb"))

model_role = pickle.load(open("model_role.pkl", "rb"))
tfidf_role = pickle.load(open("tfidf_role.pkl", "rb"))

# ---------------------------
# Preprocessing Function
# ---------------------------
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# ---------------------------
# PDF Text Extractor
# ---------------------------
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Resume Screening Dashboard", layout="wide")

st.title("📄 AI-Powered Resume Screening Tool")
st.markdown("Upload a resume in **PDF format** and the system will classify it into a job category and predict the most suitable role.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    # Extract text
    resume_text = extract_text_from_pdf(uploaded_file)

    if resume_text.strip() == "":
        st.error("⚠️ Could not extract text from this PDF. Please try another one.")
    else:
        # Preprocess
        cleaned_text = preprocess(resume_text)

        # Category Prediction
        X_cat = tfidf_category.transform([cleaned_text]).toarray()
        pred_category = model_category.predict(X_cat)[0]
        prob_category = model_category.predict_proba(X_cat).max()

        # Role Prediction
        X_role = tfidf_role.transform([cleaned_text]).toarray()
        pred_role = model_role.predict(X_role)[0]
        prob_role = model_role.predict_proba(X_role).max()

        # Display Results
        st.subheader("🔎 Prediction Results")
        st.success(f"**Category:** {pred_category}  \nConfidence: {prob_category:.2%}")
        st.success(f"**Role:** {pred_role}  \nConfidence: {prob_role:.2%}")

        # Show extracted resume text (optional)
        with st.expander("📜 View Extracted Resume Text"):
            st.write(resume_text)

        # Suggestions Section
        st.subheader("💡 Recommendations")
        st.write(f"This resume seems best suited for **{pred_role}** under **{pred_category}** category. Consider highlighting more relevant keywords to improve ATS matching.")



