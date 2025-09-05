# 📄 Resume Screening with NLP + Streamlit

This project is an **AI-powered Resume Screening System** that uses **Natural Language Processing (NLP)** techniques to classify resumes into job categories automatically.  
It also provides an **interactive Streamlit dashboard** where you can upload a PDF resume and instantly get predictions about the most suitable role.

---

## 🚀 Features
- 📂 Preprocessing of resumes with NLP (tokenization, stopword removal, TF-IDF, etc.)  
- 🤖 Machine Learning classification model trained on multiple job categories  
- 📊 Visualizations for dataset insights  
- 📑 Upload a PDF resume and get predicted category in real-time  
- 🌐 Deployed with Streamlit for a user-friendly interactive experience  

---

## 🛠️ Tech Stack
- **Python 3.12**
- **Streamlit** – Interactive dashboard  
- **scikit-learn** – ML models (Logistic Regression / Random Forest / etc.)  
- **NLTK** – NLP preprocessing  
- **PyPDF2** – Extract text from PDF resumes  
- **Pandas, NumPy, Matplotlib, Seaborn** – Data analysis & visualization  

---

## 📂 Project Structure
├── Resume_screening.py # Main Streamlit app

├── resume screening(2).py# ML file 

├── model_category.pkl # Trained ML model for category prediction

├── vectorizer.pkl # TF-IDF vectorizer

├── requirements.txt # Project dependencies

├── dataset_roles.csv # Dataset 1: Roles & resumes

├── dataset_categories.csv # Dataset 2: Categories & resumes

└── README.md # Project documentation
