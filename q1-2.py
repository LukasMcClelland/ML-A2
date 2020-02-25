
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
import winsound

trainingData = pd.read_csv("train.csv")
stopWords = [line.rstrip('\n') for line in open("stopwords.txt")]

testData = trainingData.sample(frac=0.25)
trainingData = trainingData.sample(frac=0.25)

frequency = 500  # Set Frequency To 2500 Hertz
duration = 505  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
input()
# textClassifier = Pipeline([
#     ('vect', CountVectorizer(strip_accents='ascii', stop_words=stopWords, max_features=10000)),
#     ('tfidf', TfidfTransformer()),
#     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,max_iter=5, tol=None)),
# ])

textClassifier = Pipeline([
    ('vect', CountVectorizer(strip_accents='ascii', stop_words=stopWords, max_features=10000)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
#
# textClassifier.fit(trainingData['review'], trainingData['sentiment'])
# predicted = textClassifier.predict(testData['review'])
# print(np.mean(predicted==testData['sentiment']))


parameters = {
     'vect__ngram_range': [(1, 1), (1, 2)],
     'tfidf__use_idf': (True, False),
     'clf__alpha': (1e-2, 1e-3),
 }
gs_clf = GridSearchCV(textClassifier, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(trainingData['review'], trainingData['sentiment'])

for param_name in sorted(parameters.keys()):
     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))




class LogReg:
    def __init__(self, trainingData):
      self.trainingData = trainingData

    def fit(self):
        vect = CountVectorizer(max_features=1000)
        X_train = vect.fit_transform(trainingData['review'])
        y_train = trainingData['sentiment'].to_numpy()
        features_names = vect.get_feature_names()

        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
        grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
        grid.fit(X_train, y_train)
        print("Best Cross-validation score: {:.2f}".format(grid.best_score_))