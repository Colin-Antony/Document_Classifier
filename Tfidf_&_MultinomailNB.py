from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords


stop_words = list(stopwords.words('english'))
# Load the dataset

train_data = fetch_20newsgroups(subset='train')
test_data = fetch_20newsgroups(subset='test')

# Define the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.8, min_df=2,ngram_range=(1,2))),
    ('clf', MultinomialNB(alpha=0.2, fit_prior=False))
])

# Train the classifier
print("Fitting pipeline...")
pipeline.fit(train_data.data, train_data.target)
print("Pipeline fitting completed")

# Predict the test set
print("Predicting on test data..")
predicted = pipeline.predict(test_data.data)
print("Prediction Completed.")

# Evaluate the performance
accuracy = accuracy_score(test_data.target, predicted)
cm = confusion_matrix(test_data.target, predicted)
print("Accuracy:", accuracy)
# print("Confusion Matrix:\n", cm)
print(classification_report(test_data.target predicted, target_names=test_data.target_names))

