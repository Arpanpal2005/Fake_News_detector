<h1 align="center">📰 Fake News Detector</h1>

<p align="center">
  <b>An end-to-end Fake News Detection System using NLP & Machine Learning</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NLP-green"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-blue"/>
  <img src="https://img.shields.io/badge/Streamlit-red"/>
  <img src="https://img.shields.io/badge/TF--IDF-orange"/>
  <img src="https://img.shields.io/badge/Logistic%20Regression-purple"/>
</p>

---

## 📌 Project Overview

Fake news has become a serious challenge in the digital era, spreading misinformation rapidly across social media and online platforms.

This project is an **end-to-end Fake News Detection System** that analyzes the **textual content of news articles** and predicts whether the information is **Real** or **Fake** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

The repository contains:
- 📓 A **Jupyter Notebook** showcasing the complete ML pipeline  
- 🌐 A **Streamlit Web Application** for real-time prediction  

---

## 🧠 Jupyter Notebook Work (`fakenews_det.ipynb`)

The core machine learning workflow is implemented and documented in the Jupyter Notebook.

### 🔹 Dataset
- **Fake.csv** – Collection of fake news articles  
- **True.csv** – Collection of real news articles  

---

### 🔹 Text Preprocessing
The following preprocessing steps are applied:
- Conversion of text to lowercase
- Removal of punctuation, numbers, URLs, and HTML tags
- Cleaning of noisy symbols and extra whitespace

These steps ensure the model focuses on meaningful linguistic patterns.

---

### 🔹 Feature Extraction
- **TF-IDF (Term Frequency–Inverse Document Frequency)** is used
- Converts cleaned text into numerical feature vectors
- Captures the importance of words across documents

---

### 🔹 Model Training
- **Algorithm:** Logistic Regression  
- **Classification Labels:**
  - `0` → Fake News  
  - `1` → Real News  

---

### 🔹 Model Evaluation
- Accuracy score
- Confusion matrix
- Classification report (Precision, Recall, F1-score)

---

### 🔹 Model Export
After training, the following artifacts are saved:
- `model.pkl` – Trained Logistic Regression model  
- `vectorizer.pkl` – Trained TF-IDF vectorizer  

These files are used directly by the Streamlit application.

---

## 🌐 Streamlit Web Application

The Streamlit app provides a **clean, interactive frontend** for real-time fake news detection.

### ✨ Key Features
- Paste a news article or headline
- Animated heading (Fake ↔ Real)
- Displays confidence scores for **both Real and Fake**
- Confidence-threshold handling to reduce false positives
- Clear/reset input functionality
- User-friendly and responsive UI

---

<h2>☁️ Deployment</h2>
<p>
  The <b>Fake News Detector</b> is deployed as a web application using
  <b>Streamlit Cloud</b>, enabling users to interact with the machine learning
  model through a browser in real time.
</p>

<h3>🔹 Deployment Process</h3>
<ul>
  <li>The trained machine learning model and TF-IDF vectorizer were exported from the Jupyter Notebook.</li>
  <li>A Streamlit-based frontend was developed for real-time user interaction.</li>
  <li>The complete project was pushed to GitHub.</li>
  <li>The GitHub repository was connected to <b>Streamlit Cloud</b> for deployment.</li>
</ul>

<h3>🔹 Live Application</h3>
<p>
  🔗 <b>Live Demo:</b>
  <i>https://fakenewsdetector-cjf3jdrw7obofvopm8dlnh.streamlit.app/</i>
</p>

<hr>

<hr>

<h2>👨‍💻 Author</h2>
<p>
  <b>Arpan Pal</b><br>
  <i>Machine Learning & NLP Enthusiast</i>
</p>

<ul>
  <li>Interested in Natural Language Processing and applied Machine Learning.</li>
  <li>Focused on building end-to-end ML systems with real-world usability.</li>
  <li>This project was developed as part of academic learning and hands-on experimentation.</li>
</ul>

## 🧱 Project Structure

```text
Fake_News_detector/
│
├── fakenews_det.ipynb        # Jupyter Notebook (ML pipeline)
├── fakenewsdetector.py      # Streamlit frontend
├── model.pkl                # Trained ML model
├── vectorizer.pkl           # TF-IDF vectorizer
├── requirements.txt         # Project dependencies
├── Fake.csv.zip              # Fake news dataset
├── True.csv.zip              # Real news dataset
└── README.md
