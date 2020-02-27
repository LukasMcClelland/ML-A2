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
import time

# Set this variable to True to train on entire dataset. This means we have no test set to check accuracy with. No accuracy output
# Set it to False to do a train_test_split. This means we have a test set that we can check accuracy with.
trainOnEntireDataSet = False


# Stemmer function to reduce feature space
def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


if __name__ == '__main__':
    # To allow for easier cross platform execution, check if this file is being run in a Google Colab notebook
    inColab = 'google.colab' in sys.modules

    # Import datasets
    if inColab:
        trainingData = pd.read_csv("./gdrive/My Drive/train.csv")
        stopWords = [line.rstrip('\n') for line in open("./gdrive/My Drive/stopwords.txt")]
        testData = pd.read_csv("./gdrive/My Drive/test.csv")

    else:
        trainingData = pd.read_csv("train.csv")
        stopWords = [line.rstrip('\n') for line in open("stopwords.txt")]
        testData = pd.read_csv("test.csv")

    # Adjust data sets if we're producing a file to submit to Kaggle
    if trainOnEntireDataSet:
        trainingData = trainingData.sample(frac=1)
        X_train = trainingData['review']
        y_train = trainingData['sentiment']
        X_test = testData['review']
        y_test = None

    else:
        trainingData = trainingData.sample(frac=1)
        X_train, X_test, y_train, y_test = train_test_split(trainingData['review'], trainingData['sentiment'],
                                                            train_size=0.8, test_size=0.2)

    startTime = time.time()

    stemmer = SnowballStemmer("english")
    analyzer = CountVectorizer().build_analyzer()

    # Do count vectorization
    vectorizer = CountVectorizer(strip_accents='ascii', binary=False)
    vectors_train = vectorizer.fit_transform(X_train)
    vectors_test = vectorizer.transform(X_test)

    # Calculate term-frequencies and inverse document frequencies
    tf_idf_vectorizer = TfidfVectorizer()
    vectors_train_idf = tf_idf_vectorizer.fit_transform(X_train)
    vectors_test_idf = tf_idf_vectorizer.transform(X_test)

    # Normalize data
    normalizer_train = Normalizer().fit(X=vectors_train)
    vectors_train_normalized = normalizer_train.transform(vectors_train_idf)
    vectors_test_normalized = normalizer_train.transform(vectors_test_idf)

    # Run a GridSearchCV to determine optimal hyper-parameters
    nFolds = 5
    parameters = {'C': [5, 9], 'max_iter': [1000, 10000]}
    clf = LogisticRegression()
    clf = GridSearchCV(clf, parameters, cv=nFolds, n_jobs=-1, verbose=10)
    clf.fit(vectors_train_normalized, y_train)
    print("Best score found during GridSearchVH", clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, clf.best_params_[param_name]))
    y_pred = clf.predict(vectors_test_normalized)

    print("Runtime =", time.time() - startTime, "seconds")

    # Save file if doing Kaggle submission, if not then print metrics
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

