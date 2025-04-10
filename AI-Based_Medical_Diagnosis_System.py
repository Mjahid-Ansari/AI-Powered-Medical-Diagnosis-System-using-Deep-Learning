
################# 14thh

# ✅ All-in-One AI Medical Diagnosis (Multimodal Fusion + X-ray + Report + Multiclass + EDA + Export + DB Ready)

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from sklearn.model_selection import train_test_split
from fpdf import FPDF
from datetime import datetime
from pymongo import MongoClient

# ====================== 🌈 Streamlit UI Setup ====================== #
st.set_page_config(page_title="🧠 AI Medical Diagnosis", layout="wide")

# 🌈 Theme and custom UI
st.markdown("""
    <style>
        .stApp {
            background-color: #0a192f;
            color: white;
            padding: 20px;
            border-radius: 15px;
        }
        .css-18e3th9, .css-1d391kg {
            background-color: #112240 !important;
            color: white !important;
            border-radius: 10px;
        }
        .stTextInput>div>div>input, .stTextArea textarea {
            color: white !important;
            background-color: #233554 !important;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Advanced AI Medical Diagnosis (X-ray + Report + Fusion + Multiclass)")

# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client["MedicalAI"]
collection = db["predictions"]

# =========================== 📥 Upload CSV and Show Preview ============================= #
uploaded_csv = st.file_uploader("📄 Upload Radiology Report CSV", type="csv")
tfidf, clf = None, None
patient_ids = []

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    df['patient_id'] = ['PID' + str(random.randint(10000, 99999)) for _ in range(len(df))]
    df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df['age'] = np.random.randint(25, 80, size=len(df))
    df['sex'] = np.random.choice(['Male', 'Female'], size=len(df))
    df['symptoms'] = np.random.choice([
        'Cough, Fever', 'Chest Pain, Breathlessness', 'Fatigue, Cough', 'Shortness of Breath', 'No major symptoms'
    ], size=len(df))

    st.subheader("📊 Sample Data")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📈 Label Distribution")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    df['label'].value_counts().plot(kind='bar', ax=ax, color=sns.color_palette("Set2"))
    ax.set_xlabel("Label", color='white')
    ax.set_ylabel("Count", color='white')
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#112240')
    ax.set_facecolor('#112240')
    st.pyplot(fig)

    # ======================== 📊 EDA ============================= #
    st.subheader("📌 Dataset Overview")
    st.dataframe(df.describe(include='all'))

    # =========================== 🧠 NLP Model ============================= #
    st.subheader("🧠 Report Classification (TF-IDF + Logistic Regression)")
    tfidf = TfidfVectorizer(max_features=300)
    X = tfidf.fit_transform(df['report_text'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write("✅ Accuracy:", round(acc * 100, 2), "%")

    st.subheader("📉 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(4, 2.5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm', ax=ax_cm, cbar=False)
    ax_cm.set_xlabel("Predicted", color='white')
    ax_cm.set_ylabel("Actual", color='white')
    ax_cm.tick_params(colors='white')
    fig_cm.patch.set_facecolor('#112240')
    ax_cm.set_facecolor('#112240')
    st.pyplot(fig_cm)

    st.subheader("🧾 Classification Report")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

    # =========================== 📊 Live Dashboard ============================= #
    st.subheader("📈 Live Confidence Score Dashboard")
    label_names = clf.classes_
    class_probs = clf.predict_proba(X_test[:5])
    for i in range(min(5, len(class_probs))):
        st.markdown(f"**Patient {i+1}:**")
        chart_data = pd.DataFrame({
            'Label': label_names,
            'Probability': class_probs[i]
        })
        st.bar_chart(chart_data.set_index('Label'))

# =========================== 🖼️ CNN Model for X-ray Image ============================= #
st.header("🩻 X-ray Diagnosis (CNN Binary)")

@st.cache_resource
def load_cnn_model():
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

cnn_model = load_cnn_model()
cnn_model.eval()

image_file = st.file_uploader("🖼️ Upload Chest X-ray Image", type=['png', 'jpg', 'jpeg'])
pred_label = None
if image_file:
    image = Image.open(image_file).convert('RGB')
    st.image(image, caption="Uploaded Chest X-ray", width=250)

    transform = transforms.Compose([
        transforms.Resize((124, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0)
    output = cnn_model(input_tensor)
    pred_label = torch.argmax(output, 1).item()
    st.success("✅ CNN Prediction: " + ("Normal" if pred_label == 0 else "⚠️ Abnormal"))

# =========================== 🔀 Fusion Model ============================= #
st.header("🔬 Multimodal Prediction (Fusion: X-ray + Report)")

user_text = st.text_area("✍️ Enter a Radiology Report")
if st.button("🔍 Predict from Report + X-ray"):
    if user_text.strip() != "" and tfidf and clf:
        text_pred = clf.predict(tfidf.transform([user_text]))[0]
        prob = clf.predict_proba(tfidf.transform([user_text]))[0]
        confidence = round(np.max(prob) * 100, 2)
        fusion_result = f"🧠 NLP: {text_pred} ({confidence}%)"
        if pred_label is not None:
            fusion_result += f" | 🩻 CNN: {'Normal' if pred_label == 0 else 'Abnormal'}"
        st.success("✅ Combined Prediction: " + fusion_result)

        st.progress(int(confidence))

        # 🔁 Store in MongoDB
        fusion_data = {
            "report": user_text,
            "cnn_label": "Normal" if pred_label == 0 else "Abnormal",
            "nlp_label": text_pred,
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
       ## collection.insert_one(fusion_data)
       ## st.info("📥 Prediction saved to MongoDB.")
    else:
        st.warning("⚠️ Please upload X-ray + enter report")

# =========================== 📤 Export Predictions ============================= #
if uploaded_csv and st.button("📥 Export Predictions as CSV + PDF"):
    df['predicted_label'] = clf.predict(tfidf.transform(df['report_text']))
    df.to_csv("predicted_multimodal.csv", index=False)
    st.success("✅ Exported as 'predicted_multimodal.csv'")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for i, row in df.iterrows():
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Patient ID: {row['patient_id']}", ln=True)
        pdf.cell(200, 10, txt=f"Timestamp: {row['timestamp']}", ln=True)
        pdf.cell(200, 10, txt=f"Age: {row['age']} | Sex: {row['sex']}", ln=True)
        pdf.cell(200, 10, txt=f"Symptoms: {row['symptoms']}", ln=True)
        pdf.cell(200, 10, txt=f"Report: {row['report_text'][:100]}...", ln=True)
        pdf.cell(200, 10, txt=f"Predicted Diagnosis: {row['predicted_label']}", ln=True)
    pdf.output("predicted_reports.pdf")
    st.success("📄 PDF Exported as 'predicted_reports.pdf'")


