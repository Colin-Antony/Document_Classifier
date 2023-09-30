"""Go to /docs to access Swagger UI. Easy to use fron there"""

import joblib
from Load_Dependencies import *
from fastapi import FastAPI
import uvicorn

SVM_classifier = FastAPI()


@SVM_classifier.get("/")
def index():
    return {'message': "Hello! This here is my Text classifier. Go to \\docs to use!"}


@SVM_classifier.get("/Welcome")
def greet(name: str):
    return {'Greeting': f"Hello {name}!!!"}


@SVM_classifier.post("/Text_Classifier")
def classify_text(sentence: str):
    filename = "SVM_Text_Classifier.joblib"
    model = joblib.load(filename)
    sentence = preprocess_text(sentence)
    prediction = model.predict([sentence])
    return {'prediction': target_names[prediction[0]]}


if __name__ == "__main__":
    uvicorn.run(SVM_classifier, host='127.0.0.1', port=8000)
