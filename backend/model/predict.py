import os
import pickle
from utils.text_cleaner import clean_text, contains_sensitive_word, remove_sensitive_words

# Load model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), "toxicity_model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

def predict_toxicity(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned]).toarray()
    original_prediction = model.predict(vector)[0]

 # Apply adjusted logic if it contains neutral-sensitive words
    if contains_sensitive_word(text):
        modified_message = remove_sensitive_words(cleaned)
        modified_vector = vectorizer.transform([modified_message]).toarray()
        modified_prediction = model.predict(modified_vector)[0]

# If original was toxic but modified is not => neutral content => not toxic
        if original_prediction == 1 and modified_prediction == 0:
            return 0
    return int(original_prediction)
