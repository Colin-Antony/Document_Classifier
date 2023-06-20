import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
newsgroups_data = fetch_20newsgroups(subset='all', random_state=42)
X = newsgroups_data.data
y = newsgroups_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

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
rf_pipeline = Pipeline([
    ('preprocess', CountVectorizer(preprocessor=preprocess_text,
                                   ngram_range=(1, 1), max_df=0.8, min_df=2)),
    ('tfidf', TfidfTransformer()),
    ('rf', RandomForestClassifier(n_estimators=100, max_features='sqrt')),
])

# Fit the model
print("Fitting")
rf_pipeline.fit(X_train, y_train)
print("done")


# Test the model

X_test_preprocessed = []
for text in X_test:
    preprocessed_text = preprocess_text(text)
    X_test_preprocessed.append(preprocessed_text)

print("predicting")
predicted = rf_pipeline.predict(X_test_preprocessed)
print("done")

# Print the accuracy

accuracy = accuracy_score(y_test, predicted)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, predicted,
                            target_names=newsgroups_data.target_names))

filename = 'rf_pipeline.sav'
joblib.dump(rf_pipeline, filename)

def predict_category(s, train=newsgroups_data, model=rf_pipeline):
    pred = model.predict([s])
    return train.target_names[pred[0]]
