# AI-Powered-Medical-Diagnosis-System-using-Deep-Learning


# 🧠 AI-Powered Medical Diagnosis System using Deep Learning (CNN, BERT, NLP & Multimodal Fusion)

> A real-world, full-stack AI solution that assists doctors in diagnosing chest diseases like **Tuberculosis**, **Pneumonia**, and **Pulmonary Fibrosis** using X-ray images and radiology reports.

---

## 🚀 Project Overview

This project is an advanced multimodal AI system combining **Computer Vision**, **Natural Language Processing**, and **Streamlit UI** to empower frontline healthcare workers with faster and explainable diagnostics.

🔬 **Multimodal Fusion** (Image + Text)  
🧠 **CNN (ResNet18)** for Chest X-ray classification  
📝 **BERT & TF-IDF** for radiology report understanding  
📊 **Streamlit UI** with interactive charts, prediction interface, and PDF export  
🗃️ **MongoDB** for storing patient predictions like a lightweight EMR  

---

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, ResNet18, BERT
- **NLP Models:** TF-IDF + Logistic Regression, BERT-ready pipeline
- **Frontend/UI:** Streamlit (modern, responsive)
- **Database:** MongoDB (NoSQL for storing predictions)
- **EDA/Charts:** Seaborn, Matplotlib
- **Deployment-ready:** Export to PDF/CSV, live patient tracking

---

## 💡 Real-World Use Cases

### ✅ 1. Rural Clinics Without Radiologists
> A general physician uploads an X-ray and handwritten report.  
✔️ Predicts abnormal vs normal  
✔️ Interprets the text report  
✔️ Combines both for a confident AI diagnosis

---

### ✅ 2. Emergency Room Doctors (Tier-2 Hospitals)
> Doctors upload scan + report while treating patients.  
✔️ Instant AI diagnosis with confidence scores  
✔️ MongoDB stores all predictions for audits  
✔️ Streamlit UI enables fast access and summary

---

### ✅ 3. Medical Interns & Trainees
> Students upload historical reports for learning.  
✔️ Compare AI vs actual results  
✔️ Export prediction summary as PDF  
✔️ EDA insights help learn disease pattern

---

## 🔍 Key Challenges Tackled

- ✅ CNN trained on **limited medical image data**
- ✅ Handling **noisy radiology text reports** (shorthand, misspellings)
- ✅ Building a **real-time fusion model**
- ✅ Creating **doctor-friendly, non-technical UI**
- ✅ Implementing **live database storage + PDF reports**

---

## 🖼️ Sample UI

| Upload Section | Prediction Output | Confidence Bar |
|----------------|-------------------|----------------|
| ![upload](images/upload_xray.png) | ![results](images/prediction_result.png) | ![chart](images/chart_confidence.png) |

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/ai-medical-diagnosis.git
cd ai-medical-diagnosis
pip install -r requirements.txt
streamlit run fake_news.py
