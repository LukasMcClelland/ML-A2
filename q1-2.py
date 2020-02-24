
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

trainingData = pd.read_csv("train.csv")
trainingData = trainingData.sample(frac=0.25)
testData = trainingData.sample(frac=0.25)

vectorizer = CountVectorizer(analyzer='word', max_features=5000, binary=True)
trainMatrix = vectorizer.fit_transform(trainingData['review'])
trainMatrixAsArray = trainMatrix.toarray()

testMatrix = vectorizer.transform(testData['review'])
testMatrixAsArray = trainMatrix.toarray()

X = trainMatrixAsArray
y = trainingData['sentiment']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
print("Done fitting")
print(clf.predict([testMatrixAsArray[0]]))
print(testData['sentiment'][0])

