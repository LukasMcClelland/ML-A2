import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from nltk.stem.snowball import SnowballStemmer

# Set this variable to True to train on entire dataset. This means we have no test set to check accuracy with. No accuracy output
# Set it to False to do a train_test_split. This means we have a test set that we can check accuracy with.
trainOnEntireDataSet = False

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

import time
time.time()
startTime = time.time()
print("Runtime =", time.time() - startTime, "seconds")

if __name__ == '__main__':
    # To allow for easier cross platform execution, check if this file is being run in a Google Colab notebook
    inColab = 'google.colab' in sys.modules

    if inColab:
        fractionOfDataToUse = 1
        trainingData = pd.read_csv("./gdrive/My Drive/train.csv")
        stopWords = [line.rstrip('\n') for line in open("./gdrive/My Drive/stopwords.txt")]
        testData = pd.read_csv("./gdrive/My Drive/test.csv")

    else:
        fractionOfDataToUse = 0.5
        trainingData = pd.read_csv("train.csv")
        stopWords = [line.rstrip('\n') for line in open("stopwords.txt")]
        testData = pd.read_csv("test.csv")

    trainingData = trainingData.sample(frac=fractionOfDataToUse)

    if trainOnEntireDataSet:
        X_train = trainingData['review']
        y_train = trainingData['sentiment']
        X_test = testData['review']
        y_test = None

    else:
        X_train, X_test, y_train, y_test = train_test_split(trainingData['review'], trainingData['sentiment'], train_size=0.8, test_size=0.2)

    stemmer = SnowballStemmer("english")
    analyzer = CountVectorizer().build_analyzer()

    vectorizer = CountVectorizer(strip_accents='ascii', analyzer=stemmed_words, binary=True, token_pattern='\\b[a-zA-Z]+\\b')
    vectors_train = vectorizer.fit_transform(X_train)
    vectors_test = vectorizer.transform(X_test)

    tf_idf_vectorizer = TfidfVectorizer()
    vectors_train_idf = tf_idf_vectorizer.fit_transform(X_train)
    vectors_test_idf = tf_idf_vectorizer.transform(X_test)

    normalizer_train = Normalizer().fit(X=vectors_train)
    vectors_train_normalized = normalizer_train.transform(vectors_train_idf)
    vectors_test_normalized = normalizer_train.transform(vectors_test_idf)

    nFolds = 5
    parameters = {'C': [8.75, 9, 9.25], 'max_iter': [1000, 10000]}
    clf = LogisticRegression()
    clf = GridSearchCV(clf, parameters, cv=nFolds, verbose=10)
    clf.fit(vectors_train_normalized, y_train)
    print("Best score found during GridSearchVH", clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, clf.best_params_[param_name]))
    y_pred = clf.predict(vectors_test_normalized)

    if trainOnEntireDataSet:
        if inColab:
            from google.colab import files

            with open('submission.txt', 'w') as f:
                f.write('id,sentiment\n')
                for x in range(len(y_pred)):
                    f.write(str(x) + "," + y_pred[x] + "\n")
        else:
            with open('submission.csv', 'w') as f:
                f.write('id,sentiment\n')
                for x in range(len(y_pred)):
                    f.write(str(x) + "," + y_pred[x] + "\n")
    else:
        print("Report:\n", metrics.classification_report(y_test, y_pred))

