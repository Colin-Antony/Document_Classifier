import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load the dataset
newsgroups_data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
X = newsgroups_data.data
y = newsgroups_data.target

# Preprocess the data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I | re.A)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    text = ' '.join(tokens)
    return text

X_preprocessed = []
for text in X:
    preprocessed_text = preprocess_text(text)
    X_preprocessed.append(preprocessed_text)

# Convert the data to a matrix of token counts
vectorizer = CountVectorizer(stop_words='english', min_df=2)
X_counts = vectorizer.fit_transform(X_preprocessed)

# Convert the token count matrix to a numpy array
X_array = X_counts.toarray()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_array, y, test_size=0.3, random_state=42)

# Convert the labels to one-hot encoded vectors
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Define the neural network model
input_dim = X_train.shape[1]
num_classes = y_train.shape[1]
model = Sequential()
model.add(Dense(512, input_dim=input_dim, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
batch_size = 10
epochs = 10
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test_classes, y_pred_classes, target_names=newsgroups_data.target_names))
