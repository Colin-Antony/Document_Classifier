import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report, accuracy_score

# Fetching the data and splitting
newsgroups_data = fetch_20newsgroups(subset='all', random_state=42)
X = newsgroups_data.data
y = newsgroups_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# NLP preprocessing steps
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


# Defining the pipeline
svm_classifier = Pipeline([
    ("Preprocess_Tfidf", TfidfVectorizer(preprocessor=preprocess_text,
                                         ngram_range=(1,2), max_df=0.75)),
    ("SVM", SVC(kernel="linear", C=10))
])

# Fitting the pipeline
print("Fitting the pipeline")
svm_classifier.fit(X_train,y_train)
print("Fitting completed")


# Predicting the test set
print("Predicting test data")
predicted = svm_classifier.predict(X_test)
print("prediction completed")

accuracy = accuracy_score(y_test, predicted)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, predicted,
                            target_names=newsgroups_data.target_names))
