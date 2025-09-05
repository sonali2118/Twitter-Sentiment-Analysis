# 🐦 Twitter Sentiment Analysis

This project performs **sentiment analysis** on Twitter text data to classify tweets into categories such as **Positive, Negative, Neutral, and Irrelevant**.  
It leverages **NLP preprocessing**, **TF-IDF vectorization**, and **Machine Learning models** to predict sentiment.

---

## 📌 Project Overview
- Dataset size: **~75k tweets**
- Target variable: `sentiment`
- Classes:
  - Positive
  - Negative
  - Neutral
  - Irrelevant
- Key Tasks:
  - Text cleaning & preprocessing
  - Feature extraction with **TF-IDF**
  - Training **Random Forest Classifier**
  - Model evaluation

---

## ⚙️ Tech Stack
- **Python 3.x**
- **Libraries**:
  - pandas, numpy
  - matplotlib, seaborn, wordcloud
  - scikit-learn
  - nltk
  - preprocess_kgptalkie

---

## 📂 Project Structure
Twitter-Sentiment-Analysis/
│-- data/
│ └── tweets.csv
│-- notebooks/
│ └── sentiment_analysis.ipynb
│-- images/
│ └── wordcloud_positive.png
│ └── confusion_matrix.png
│-- README.md


---

## 🔍 Data Preprocessing
1. Lowercasing text  
2. Removing URLs, HTML tags, special characters, stopwords  
3. Tokenization & Lemmatization  
4. Generating WordClouds for visualization  

---

## 📊 Model Training
- **Vectorizer**: `TfidfVectorizer`
- **Model**: `RandomForestClassifier`
- **Pipeline**: TF-IDF + Classifier  

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

clf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
])

clf.fit(X_train, y_train)
