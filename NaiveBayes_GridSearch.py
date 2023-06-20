from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV


# Load the 20 Newsgroups dataset

newsgroups_train = fetch_20newsgroups(subset='train', shuffle="True")
newsgroups_test = fetch_20newsgroups(subset='test', shuffle="True")

# Define the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

# Define the hyperparameters to tune
params = {
    'tfidf__max_features': [5000, 10000],
    'tfidf__stop_words': [None, 'english'],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'nb__alpha': [0.1, 0.5, 1.0]
}

# Perform a grid search over the hyperparameters
grid_search = GridSearchCV(pipeline, params, cv=5)
grid_search.fit(newsgroups_train.data, newsgroups_train.target)

# Print the best parameters and accuracy score
print("Best Parameters:", grid_search.best_params_)
print("Accuracy Score:", grid_search.best_score_)

# Evaluate the classifier on the test set
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(newsgroups_test.data)
accuracy = (y_pred == newsgroups_test.target).mean()
print("Test Accuracy:", accuracy)
