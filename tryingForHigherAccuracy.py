

# This file currently only runs in colab. You'll have to link up Colab and your google drive and make sure that files:
# train.csv, test.csv
# are in the root folder of your gdrive

# I recommend trying different sklearn classifiers (and playing with the parameters you can pass to each one)
# There's way of manipulating the classifiers and manipulating the CountVectorizer/Normalizer/TfidfVectorizer (as you can see from the messy code)


# Honestly, I was kinda just trying different combination of things and hoping they gave better accuracy. The CV that gave me
# the highest score is the one that's currently NOT commented out. I've recorded the accuracies of my attempts.

# I know it seems overwhelming but it's honestly not that bad. Just some reading documentation :)


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn import linear_model
from sklearn.svm import LinearSVC

trainingData = pd.read_csv("./gdrive/My Drive/train.csv")
stopWords = [line.rstrip('\n') for line in open("./gdrive/My Drive/stopwords.txt")]
stopWords2 = set(stopwords.words('english'))
testData = pd.read_csv("./gdrive/My Drive/test.csv")

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


stemmer = SnowballStemmer("english")
analyzer = CountVectorizer().build_analyzer()


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


# stem_vectorizer = CountVectorizer(analyzer=stemmed_words)

# trainingData = trainingData.sample(frac=1)

X_train = trainingData['review']
y_train = trainingData['sentiment']

X_test = testData['review']
# y_test is unknown (save y_preds to file)

# X_train, X_test, y_train, y_test = train_test_split(trainingData['review'], trainingData['sentiment'], train_size=0.8, test_size=0.2)

# 88.5
# vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True, analyzer='word', token_pattern='\\b[a-zA-Z]+\\b|\\b\\w+\\/[1-9]+\\b|\\b[1-9]+\\/\\w+\\b', ngram_range=(1,2), binary=False)

# 87.1
# vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True, analyzer='word', token_pattern='\\b[a-zA-Z]+\\b|\\b\\w+\\/[1-9]+\\b|\\b[1-9]+\\/\\w+\\b', ngram_range=(1,2), binary=True)

# 87.7
# vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True, analyzer='word', ngram_range=(1,2), binary=False)

# 88.6
# vectorizer = CountVectorizer(strip_accents='ascii',  binary=True)

#
# vectorizer = CountVectorizer(strip_accents='ascii', token_pattern='\\b[a-zA-Z]+\\b', stop_words=stopWords, binary=False)

# 87.6%
# vectorizer = CountVectorizer(strip_accents='ascii',  ngram_range=(2,3),   binary=False)

# ~88.2%
# vectorizer = CountVectorizer(strip_accents='ascii',  ngram_range=(1,2),   binary=False)

# 87.9
# vectorizer = CountVectorizer(strip_accents='ascii', ngram_range=(1,3), binary=False)

# 88
# vectorizer = CountVectorizer(strip_accents='ascii', ngram_range=(2,3), binary=False)

# 88.1
# vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words=stopWords2)

# PRETTY GOOD on fold got 89 (87.2)
# vectorizer = CountVectorizer(analyzer=stemmed_words)

# 87.8
# vectorizer = CountVectorizer(strip_accents='ascii', analyzer=stemmed_words)

# 88.3
# vectorizer = CountVectorizer(strip_accents='ascii', analyzer=stemmed_words, binary=True)

# 88.5
vectorizer = CountVectorizer(strip_accents='ascii', analyzer=stemmed_words, binary=True,
                             token_pattern='\\b[a-zA-Z]+\\b')

# 88.25
# vectorizer = CountVectorizer(strip_accents='ascii', analyzer=stemmed_words, binary=True, token_pattern='\\b[a-zA-Z]+\\b|\\b\\w+\\/[1-9]+\\b|\\b[1-9]+\\/\\w+\\b')

# ? 87
# vectorizer = CountVectorizer(strip_accents='ascii', analyzer=stemmed_words, binary=True, token_pattern='\\b[a-zA-Z]+\\b|\\b\\w+\\/[1-9]+\\b|\\b[1-9]+\\/\\w+\\b', ngram_range=(1,3))


# 87
# vectorizer = CountVectorizer(strip_accents='ascii', analyzer=stemmed_words, binary=True, token_pattern='\\b[a-zA-Z]+\\b', ngram_range=(1,3))

# 87%
# vectorizer = CountVectorizer(strip_accents='ascii', analyzer=stemmed_words, binary=True, max_df = 0.5, min_df=0.25)

# 88.1
# vectorizer = CountVectorizer(strip_accents='ascii', analyzer=stemmed_words, binary=False)

# 87.6
# vectorizer = CountVectorizer(strip_accents='ascii', analyzer=stemmed_words, binary=False, ngram_range=(1,2))

# 88.2
# vectorizer = CountVectorizer(strip_accents='ascii', analyzer=stemmed_words, binary=True, ngram_range=(1,2))


# vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True, analyzer='word', ngram_range=(1,3), binary=False)
vectors_train = vectorizer.fit_transform(X_train)
vectors_test = vectorizer.transform(X_test)

tf_idf_vectorizer = TfidfVectorizer()
vectors_train_idf = tf_idf_vectorizer.fit_transform(X_train)
vectors_test_idf = tf_idf_vectorizer.transform(X_test)

normalizer_train = Normalizer().fit(X=vectors_train)
vectors_train_normalized = normalizer_train.transform(vectors_train_idf)
vectors_test_normalized = normalizer_train.transform(vectors_test_idf)

nFolds = 5
parameters = {'C': [8.75, 9, 9.25], 'max_iter': [10000, 100000]}
clf = LogisticRegression()
clf = GridSearchCV(clf, parameters, cv=nFolds, verbose=10)
clf.fit(vectors_train_normalized, y_train)
y_pred = clf.predict(vectors_test_normalized)

from google.colab import files

with open('submission.csv', 'w') as f:
    f.write('id,sentiment\n')
    for x in range(len(y_pred)):
        f.write(str(x) + "," + y_pred[x] + "\n")


# I'm storing random shit in this function so I can easily decreased the size of the file on screen.
# There might be some useful syntax things in here
def iMadeThisAFunctionSoICouldFoldTheCode():
    # clf = LogisticRegression()
    # clf.fit(vectors_train_normalized, y_train)
    # y_pred = clf.predict(vectors_test_normalized)
    # print(metrics.accuracy_score(y_test, y_pred))

    # kf = KFold(n_splits=5)
    # for train_index, test_index in kf.split(trainingData['review']):
    #     X = trainingData['review']
    #     y = trainingData['sentiment']
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #     textClassifier = Pipeline([
    #         ('vect', CountVectorizer()),
    #         ('tfidf', TfidfVectorizer()),
    #         ('nrml', Normalizer()),
    #         ('clf', LogisticRegression()),
    #     ])

    #     parameters = {
    #         'vect__strip_accents': ['ascii'],
    #         'vect__analyzer': ['word'],
    #         'vect__token_pattern': ['\\b[a-zA-Z]+\\b|\\b\\w+\\/[1-9]+\\b|\\b[1-9]+\\/\\w+\\b'],
    #         'vect__stop_words': [stopWords],
    #         'vect__max_features': [5000],
    #         'vect__binary': [False, True],
    #         'vect__ngram_range': [(1, 2)],
    #         # 'clf__alpha': ([1, 1e-1, 1e-2, 1e-3]),
    #     }
    #     gs_clf = GridSearchCV(textClassifier, parameters, cv=5, refit=False)
    #     gs_clf = gs_clf.fit(vectors_train_normalized, y_train)

    #     # print(gs_clf.best_score_)
    #     # for param_name in sorted(parameters.keys()):
    #     #      print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    #     best = gs_clf.best_params_
    #     textClassifier = Pipeline([
    #         ('vect', CountVectorizer(strip_accents=best['vect__strip_accents'],
    #                                 token_pattern=best['vect__token_pattern'],
    #                                 stop_words=best['vect__stop_words'],
    #                                 max_features=best['vect__max_features'],
    #                                 ngram_range=best['vect__ngram_range'])),
    #         ('tfidf', TfidfVectorizer()),
    #         ('nrml', Normalizer()),
    #         ('clf', LogisticRegression()),
    #     ])

    #     textClassifier.fit(vectors_train_normalized, y_train)
    #     predicted = textClassifier.predict(vectors_test_normalized)
    #     print(metrics.accuracy_score(y_test, y_pred))
    #     print(textClassifier['vect'].get_feature_names())
    pass
