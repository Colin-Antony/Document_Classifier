import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import  cross_val_score
import joblib


# Load dataset
newsgroups_data = fetch_20newsgroups(subset='all', random_state=42)
X = newsgroups_data.data
y = newsgroups_data.target

# Define preprocessing steps
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I | re.A)
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    tokens = [t for t in tokens if t not in stop_words]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    # Rejoin tokens into a string
    text = ' '.join(tokens)
    return text


# Define the pipeline
nb_pipeline = Pipeline([
    ('preprocess', CountVectorizer(preprocessor=preprocess_text,
                                   ngram_range=(1, 1), max_df=0.8, min_df=2)),
    ('tfidf', TfidfTransformer()),
    ('nb', MultinomialNB(alpha=0.1)),
])

# Cross-validation
scores = cross_val_score(nb_pipeline, X, y, cv=5)

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean()}")

