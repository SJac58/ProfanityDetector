import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# === Define Sensitive-Neutral Words ===
neutral_sensitive_words = {
    "girl", "boy", "boys", "woman", "man", "lady", "gentleman", "guy", "dude",
    "child", "little", "baby", "teen", "youth", "young", "senior", "elder",
    "asian", "white", "black", "indian", "latino", "american", "european", "african",
    "blind", "deaf", "disabled", "autistic", "mental", "special", "dyslexic",
    "fat", "thin", "short", "tall", "dark", "fair",
    "mother", "father", "dad", "mom", "sister", "brother", "uncle", "aunt"
}

# === Clean Text ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(word for word in text.split() if word not in stop_words)

# === Utility: Check if text contains sensitive-neutral words ===
def contains_sensitive_word(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return any(word in neutral_sensitive_words for word in words)

# === Utility: Remove neutral-sensitive words from text ===
def remove_sensitive_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return ' '.join(word for word in words if word not in neutral_sensitive_words)
