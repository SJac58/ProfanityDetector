import pandas as pd
import re
import nltk
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

# === Setup ===
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# === Define sensitive-neutral word list ===
neutral_sensitive_words = {
    "girl", "boy", "boys", "woman", "man", "lady", "gentleman", "guy", "dude",
    "child", "little", "baby", "teen", "youth", "young", "senior", "elder",
    "asian", "white", "black", "indian", "latino", "american", "european", "african",
    "blind", "deaf", "disabled", "autistic", "mental", "special", "dyslexic",
    "fat", "thin", "short", "tall", "dark", "fair",
    "mother", "father", "dad", "mom", "sister", "brother", "uncle", "aunt"
}

# === 1. Load Dataset ===
df = pd.read_csv(r'E:\VSCodeProj\ProfanityDetectorModule\backend\dataset\train.csv')

# === 2. Prepare Labels ===
df['toxic'] = df['class'].apply(lambda x: 1 if x in [0, 1] else 0)

# === 3. Clean Text ===
def clean_text(text):
    if pd.isnull(text):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['clean_tweet'] = df['tweet'].apply(clean_text)

# === 4. Main Model: TF-IDF + SVM ===
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_tweet']).toarray()
y = df['toxic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearSVC()
model.fit(X_train, y_train)

print("✅ Main Model Accuracy:")
print("Training:", model.score(X_train, y_train))
print("Testing:", model.score(X_test, y_test))

# === 5. Second Filter Model (Modified) ===

# Function to remove neutral-sensitive words from the text
def remove_neutral_words(text):
    return ' '.join([word for word in text.split() if word not in neutral_sensitive_words])

# Function to check if the text contains neutral-sensitive words
def contains_sensitive_word(text):
    if pd.isnull(text):
        return False
    return any(word in text.lower() for word in neutral_sensitive_words)

def classify_with_filter(text):
    if contains_sensitive_word(text):
        # Remove neutral-sensitive words
        cleaned_text_without_sensitive_words = remove_neutral_words(text)
        
        # Vectorize both the original text and the modified text without sensitive words
        X_original = tfidf.transform([text]).toarray()
        X_modified = tfidf.transform([cleaned_text_without_sensitive_words]).toarray()

        # Predict toxicity with both versions
        original_prediction = model.predict(X_original)[0]
        modified_prediction = model.predict(X_modified)[0]

        # If the modified version would be toxic, treat the original as non-toxic
        if modified_prediction == 1:
            return 0  # non-toxic
        else:
            return original_prediction  # Keep the original classification
    else:
        # No sensitive words, use the original classification
        X_original = tfidf.transform([text]).toarray()
        return model.predict(X_original)[0]

# Test the filter with the dataset
df['adjusted_toxicity'] = df['clean_tweet'].apply(classify_with_filter)

# Calculate accuracy of the adjusted model
print("✅ Adjusted Model Accuracy:")
print("Adjusted Testing Accuracy:", (df['adjusted_toxicity'] == df['toxic']).mean())

# === 6. Save Models ===
os.makedirs("model", exist_ok=True)

with open("model/toxicity_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ All models and vectorizers saved to model/")
