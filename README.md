# Document_Classifier
Compared different models' accuracy in Document Classification.

A Project that involved testing the performance of various models in Text Classification, in this case, on the 20 Newsgroup Dataset.
Succesfully implemented various models and spent time optimizing their parameters to get a low amount of overfitting.
The models that were implemented are:
  i) Multinomial Naive Bayes
 ii) Support Vector Machine (SVM)
iii) Random Forest Classifier
 iv) Logistic Regression


 # Conclusions Made from the results
 In The order of computational complexity, i.e., training time from lowest to highest is as follows:
o	Naïve Bayes
o	Logistic Regression
o	Random Forest Classifier
o	SVM

 The order of accuracy that I have achieved from highest to lowest is as follows:
o	SVM
o	Naïve Bayes
o	Logistic Regression 
o	Random Forest Classifier (almost same as LR)

Therefore, the best model to use for text classification is SVM. However, if you prefer shorter computational times while training the model, use Naïve Bayes.

For a much more detailed report, go to: Document Classifers.pdf

# File Names
The files that contain the final models are:
NBwithNLP.py
SVMFinal.py
LogisticRegression.py
Random Forest.py

The rest are debug/gridsearch files.
