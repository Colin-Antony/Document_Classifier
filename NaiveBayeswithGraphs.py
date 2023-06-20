import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

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


# list of alpha values
alpha_values = np.linspace(0, 2, num=11)

# Initialize lists to store the accuracy values
train_acc = []
test_acc = []

# Define the pipeline
for alpha in alpha_values:
    nb_pipeline = Pipeline([
        ('preprocess', CountVectorizer(preprocessor=preprocess_text,
                                       ngram_range=(1, 1), max_df=0.8, min_df=2)),
        ('tfidf', TfidfTransformer()),
        ('nb', MultinomialNB(alpha=alpha)),
    ])

    # Fit the model
    print("Fitting")
    nb_pipeline.fit(X_train, y_train)
    print("done")

    # calculate accuracy on training set
    train_predicted = nb_pipeline.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predicted)
    train_acc.append(train_accuracy)

    # calculate on test set
    X_test_preprocessed = []
    for text in X_test:
        preprocessed_text = preprocess_text(text)
        X_test_preprocessed.append(preprocessed_text)
    test_predict = nb_pipeline.predict(X_test_preprocessed)
    test_accuracy = accuracy_score(y_test, test_predict)
    test_acc.append(test_accuracy)


# Plot the accuracy values
plt.plot(alpha_values, train_acc, '-o',  label='Training Set')
plt.plot(alpha_values, test_acc, '-o', label='Test Set')
plt.xlabel('Alpha Values')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Alpha Values for Multinomial Naive Bayes Model')
plt.legend()
plt.show()



