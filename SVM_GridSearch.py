# Import libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Load training and testing datasets
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)


# NLP preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I|re.A)
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    tokens = [t for t in tokens if t not in stop_words]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    # Rejoin tokens into a string
    text = ' '.join(tokens)
    return text


# Define pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(preprocessor=preprocess_text)),
    ('clf', svm.SVC())
])

# Define parameter grid for grid search
param_grid = {
    'tfidf__max_df': [0.5, 0.75, 1.0],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__kernel': ['linear', 'rbf'],
    'clf__C': [0.1, 1, 10]
}

# Perform grid search
print("start grid search")
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=3, n_jobs=3)
grid_search.fit(newsgroups_train.data, newsgroups_train.target)
print("Done")

# Print results of grid search
print("Best score:", grid_search.best_score_)
print("Best parameters:", grid_search.best_params_)

# Evaluate performance on testing dataset
print("predicting")
y_pred = grid_search.predict(newsgroups_test.data)
print(classification_report(newsgroups_test.target, y_pred, target_names=newsgroups_test.target_names))
print("done")