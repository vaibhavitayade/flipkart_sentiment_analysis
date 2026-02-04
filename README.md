# Flipkart Product Review Sentiment Analysis

## 📌 Project Overview
This project focuses on **Sentiment Analysis of real-time Flipkart product reviews**.  
The goal is to classify customer reviews as **Positive** or **Negative** and understand customer pain points using Natural Language Processing (NLP) and Machine Learning techniques.

The trained model is deployed as a **Streamlit web application** and hosted on an **AWS EC2 instance** for real-time inference.

---

## 🎯 Objective
- Classify Flipkart product reviews into **Positive** and **Negative**
- Analyze customer feedback to identify dissatisfaction reasons
- Deploy the trained model as a web application
- Make the application accessible via AWS EC2

---

## 📊 Dataset Description
- Source: Flipkart product reviews (provided dataset)
- Total Reviews: **8,518**
- Cleaned Reviews Used: **8,510**
- Key Features:
  - Reviewer Name
  - Review Title
  - Review Text
  - Ratings
  - Place of Review
  - Votes (Up / Down)
  - Month

---

## 🧹 Data Preprocessing
The dataset required minor cleaning before model training:
- Removed duplicate records
- Removed rows with missing review text
- Handled missing month values
- Standardized column names
- Created sentiment labels from ratings

**Final cleaned dataset shape:** `(8510, 8)`

---

## 🔍 Text Processing Steps
- Lowercasing
- Removal of punctuation and numbers
- Stopword removal
- Lemmatization using NLTK

---

## 🧠 Feature Engineering
- **TF-IDF (Term Frequency–Inverse Document Frequency)**
- Unigrams and bigrams
- Maximum features: 5000

---

## 🤖 Model Training
- Algorithm: **Logistic Regression**
- Train/Test Split: **80% / 20%**
- Evaluation Metric: **F1-Score**

### ✔ Performance
- Achieved strong F1-score for binary sentiment classification
- Balanced performance on positive and negative classes

---

## 🌐 Web Application
A **Streamlit-based web application** was developed where users can:
- Enter a product review
- Get instant sentiment prediction (Positive / Negative)

---

## ☁️ AWS Deployment
- Platform: **AWS EC2 (Ubuntu)**
- App Framework: **Streamlit**
- Model integration using pickle files

### 🔗 Live Deployment
