import re
import string
import spacy
from nltk.corpus import stopwords

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load NLTK stopwords
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """
    Cleans text by removing unnecessary whitespace, HTML tags, and special characters.
    """
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    return text

def normalize_text(text):
    """
    Normalizes text by converting to lowercase.
    """
    return text.lower()

def tokenize_and_remove_stopwords(text):
    """
    Tokenizes text and removes stopwords using SpaCy and NLTK.
    """
    doc = nlp(text)
    tokens = [token.text for token in doc if token.text.lower() not in stop_words and not token.is_punct]
    return " ".join(tokens)

def preprocess_text(text):
    """
    Combines all preprocessing steps.
    """
    text = clean_text(text)
    text = normalize_text(text)
    text = tokenize_and_remove_stopwords(text)
    return text

def preprocess_dataset(input_path, output_path):
    """
    Reads a CSV file, preprocesses the 'title' column, and saves the cleaned data.
    """
    import pandas as pd

    df = pd.read_csv(input_path)
    df["cleaned_title"] = df["title"].apply(preprocess_text)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
