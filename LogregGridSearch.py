import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Load dataset
newsgroups_data = fetch_20newsgroups(subset='train', random_state=1)
X = newsgroups_data.data
y = newsgroups_data.target
# newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True)

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.4, random_state=1)

# Define preprocessing steps
stop_words = set(stopwords.words('english'))
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


# Define the pipeline
logreg_pipeline = Pipeline([
    ('preprocess', CountVectorizer(preprocessor=preprocess_text, ngram_range=(1, 1), max_df=0.8, min_df=2)),
    ('tfidf', TfidfTransformer()),
    ('logreg', LogisticRegression(max_iter=1000)),
])

param_grid = {
    'logreg__penalty': ['l1', 'l2'],
    'logreg__C': [0.01, 0.1, 1, 10],
    'logreg__solver': ['liblinear', 'saga'],
    'tfidf__use_idf': [True, False]
}

# Fit the model
print("Fitting")
grid_search = GridSearchCV(logreg_pipeline, param_grid=param_grid, cv=5, verbose=1)
grid_search.fit(X_train,y_train)
# logreg_pipeline.fit(X_train, y_train)
# pred = logreg_pipeline.predict(X_train)
# print("Training acc: ",accuracy_score(y_train,pred))

print("done")
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=newsgroups_data.target_names))
