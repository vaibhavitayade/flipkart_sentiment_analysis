import pandas as pd
import re
import string
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

nltk.download('stopwords')
nltk.download('wordnet')

# =========================
# Load Data
# =========================
df = pd.read_csv("cleaned_data.csv")

# Normalize column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

print("\nCOLUMNS IN DATASET:")
print(df.columns)

# =========================
# Detect rating column
# =========================
rating_col = None
for col in df.columns:
    if "rating" in col:
        rating_col = col
        break

if rating_col is None:
    raise Exception(" Rating column not found in dataset")

print(f"\nUsing rating column: {rating_col}")

# =========================
# Keep required columns
# =========================
df = df[['review_text', rating_col]]
df.rename(columns={rating_col: 'rating'}, inplace=True)

# Remove neutral reviews
df = df[df['rating'] != 3]

# Create sentiment
df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

# =========================
# Text Cleaning
# =========================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

df['clean_review'] = df['review_text'].apply(clean_text)

# =========================
# TF-IDF
# =========================
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df['clean_review'])
y = df['sentiment']

# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Train Model
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test)
print("\nF1 Score:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================
# Save Model
# =========================
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")
