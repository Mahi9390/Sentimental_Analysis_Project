
import re
import string
import emoji
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from deep_translator import GoogleTranslator
from sklearn.base import BaseEstimator, TransformerMixin

# NLTK Data Download
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def translate_text(self, text):
        try:
            result = GoogleTranslator(source='auto', target='en').translate(text)
            return result
        except Exception as e:
            return text

    def preprocess_single_text(self, text):
        if not isinstance(text, str):
            text = str(text)

        text = text.lower()
        text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
        text = re.sub(r'[@#]\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'\s+', ' ', text).strip()

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.apply(self.preprocess_single_text)
        elif isinstance(X, pd.DataFrame):
            X_copy = X.copy()
            if 'title' in X_copy.columns:
                X_copy['title'] = X_copy['title'].apply(self.preprocess_single_text)
            if 'body' in X_copy.columns:
                X_copy['body'] = X_copy['body'].apply(self.preprocess_single_text)
            return X_copy
        else:
            raise TypeError("Expected pandas Series or DataFrame for transform method")
