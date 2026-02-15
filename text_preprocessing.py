# text_preprocessing.py

import re
import string
import emoji
import nltk

from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin


# ------------------------------------------------------------
# Ensure required NLTK resources are available
# ------------------------------------------------------------
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)


# ------------------------------------------------------------
# Translation Function
# ------------------------------------------------------------
def translate_text(text):
    if not isinstance(text, str):
        text = str(text)

    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text


# ------------------------------------------------------------
# Text Cleaning Function
# ------------------------------------------------------------
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    # Remove URLs
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)

    # Remove @mentions and #hashtags
    text = re.sub(r'[@#]\w+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


# ------------------------------------------------------------
# Custom Transformer (if used inside Pipeline)
# ------------------------------------------------------------
class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer for text preprocessing.
    Use this inside your Pipeline if required.
    """

    def __init__(self, translate=False):
        self.translate = translate

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed = []
        for text in X:
            if self.translate:
                text = translate_text(text)
            processed.append(preprocess_text(text))
        return processed
