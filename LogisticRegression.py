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
newsgroups_data = fetch_20newsgroups(subset='all', random_state=1)
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
    ('preprocess', CountVectorizer(preprocessor=preprocess_text,
                                   ngram_range=(1, 1), max_df=0.8, min_df=2)),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('logreg', LogisticRegression(max_iter=1000, penalty='l2',
                                  solver='liblinear', C=10)),
])


# Fit the model
print("Fitting")
logreg_pipeline.fit(X_train, y_train)
print("done")

X_test_preprocessed = []
for text in X_test:
    preprocessed_text = preprocess_text(text)
    X_test_preprocessed.append(preprocessed_text)

# Evaluate the model
predict = logreg_pipeline.predict(X_test_preprocessed)
print("Accuracy:", accuracy_score(y_test, predict))
print(classification_report(y_test, predict,
                            target_names=newsgroups_data.target_names))
