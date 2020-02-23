# TODO
#  maybe use TF-IDF features with sklearn (TfidfVectorizer or TfidfTransformer)
#  https://towardsdatascience.com/hacking-scikit-learns-vectorizers-9ef26a7170af   might have ways of improving results

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
import sys

maxFeatures = 5000
inColab = False  # No need to change this, it's checked automatically :)


# CURRENTLY NOT USED - this function was from previous stuff I was testing. Might come in handy later
# Split data up, return a dictionary where keys are class names and values are lists of entries
def separateClasses(data, labelVector):
    copyVector = labelVector.to_numpy()
    tempDict = dict()
    tempDict['positive'] = list()
    tempDict['negative'] = list()
    for x in range(len(data)):
        if copyVector[x] == 'positive':
            tempDict['positive'].append(data[x])
        else:
            tempDict['negative'].append(data[x])
    if inColab:
        tempDict['positive'] = np.array(tempDict['positive'], dtype=np.uint64)
        tempDict['negative'] = np.array(tempDict['negative'], dtype=np.uint64)
    else:
        tempDict['positive'] = np.array(tempDict['positive'], dtype=np.uint32)
        tempDict['negative'] = np.array(tempDict['negative'], dtype=np.uint32)
    return tempDict


stemmer = PorterStemmer()


class CountVectorizerWithStemmer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizerWithStemmer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


def train(trainingData):
    # TODO
    # stemming maybe, words in first sentence get counted twice (upweighting)?

    # 0 - 84.0%
    # vectorizer = CountVectorizer(analyzer='word', stop_words=stopWords, max_features=maxFeatures, binary=False)

    # 1 - 87.0%
    # vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True, analyzer='word', token_pattern='[a-zA-Z]+', stop_words=stopWords, max_features=maxFeatures, binary=False)

    # 2 - 87.4%
    vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True, analyzer='word',
                                 token_pattern='(?u)[a-zA-Z]+|\\b\\w\\/\\w+\\b', stop_words=stopWords,
                                 max_features=maxFeatures, binary=True)

    # 3 - 85.3%
    # vectorizer = CountVectorizerWithStemmer(strip_accents='ascii', lowercase=True, analyzer='word', token_pattern='(?u)[a-zA-Z]+|\\b\\w\\/\\w+\\b', stop_words=stopWords, max_features=maxFeatures, binary=False)

    # 4 - 86.1%
    # vectorizer = CountVectorizerWithStemmer(strip_accents='ascii', lowercase=True, analyzer='word', token_pattern='(?u)[a-zA-Z]+', stop_words=stopWords, max_features=maxFeatures, binary=False)

    countMatrix = vectorizer.fit_transform(trainingData['review'])
    countMatrixAsArray = countMatrix.toarray()
    print("Show the", maxFeatures, "most popular words in the corpus:\n", vectorizer.get_feature_names())
    print("\nNumber of reviews (rows) in the word count table:", len(countMatrixAsArray))
    print("Number of words   (cols) in the word count table:", len(countMatrixAsArray[0]))
    print("\nThe word count table:\n", countMatrixAsArray, '\n')
    numWords = len(countMatrixAsArray[0])

    # Create dictionary with separated classes
    separated = separateClasses(countMatrixAsArray, trainingData['sentiment'])
    totalNegReviews = len(separated['negative'])
    totalPosReviews = len(separated['positive'])
    totalReviews = totalNegReviews + totalPosReviews
    theta0 = totalNegReviews / totalReviews
    theta1 = totalPosReviews / totalReviews

    negCounts = separated['negative'].sum(axis=0)
    posCounts = separated['positive'].sum(axis=0)

    if inColab:
        theta_j_0 = np.array([0 for _ in range(numWords)], dtype='float32')
        theta_j_1 = np.array([0 for _ in range(numWords)], dtype='float32')
    else:
        theta_j_0 = np.array([0 for _ in range(numWords)], dtype='float32')
        theta_j_1 = np.array([0 for _ in range(numWords)], dtype='float32')

    for x in range(numWords):
        theta_j_0[x] = (negCounts[x] + 1) / (totalNegReviews + numWords)
        theta_j_1[x] = (posCounts[x] + 1) / (totalPosReviews + numWords)

    return numWords, vectorizer, theta0, theta1, theta_j_0, theta_j_1


def test(testData, theta0, theta1, theta_j_0, theta_j_1):
    print("Start testing...", end='')
    transformedData = vectorizer.transform(testData['review'])
    dataToArray = transformedData.toarray()
    labelVector = testData['sentiment'].to_numpy()

    # firstLogTerm = np.log(np.true_divide(theta_j_1, theta_j_0))
    # secondLogTerm = np.log(np.true_divide(np.subtract(np.ones(len(theta_j_1)), theta_j_1), np.subtract(np.ones(len(theta_j_0)), theta_j_0)))
    firstTerm = np.dot(dataToArray, np.log(np.true_divide(theta_j_1, theta_j_0)))
    secondTerm = np.dot(np.subtract(np.ones((len(dataToArray), len(dataToArray[0]))), dataToArray),
                        np.log(np.true_divide(np.subtract(np.ones(len(theta_j_1)), theta_j_1),
                                              np.subtract(np.ones(len(theta_j_0)), theta_j_0))))
    fastLogLikelihoods = np.add(np.add(firstTerm, secondTerm), np.log(theta1 / theta0))

    correctPredictions = 0
    for i in range(len(testData)):
        if fastLogLikelihoods[i] > 0 and labelVector[i] == 'positive':
            correctPredictions += 1
        if fastLogLikelihoods[i] < 0 and labelVector[i] == 'negative':
            correctPredictions += 1

    print("DONE")
    print("Accuracy = {0:.1f}%".format(100 * correctPredictions / len(labelVector)))


if __name__ == "__main__":
    inColab = 'google.colab' in sys.modules
    if inColab:
        print("Colab environment detected. Running program in high RAM usage mode\n")
        maxFeatures = 10000
        trainingData = pd.read_csv("./gdrive/My Drive/train.csv")  # | review (text)  | sentiment (pos/neg) |
        # testData = pd.read_csv("test.csv")       # | id (review id) |   review (text)     |
        stopWords = [line.rstrip('\n') for line in open("./gdrive/My Drive/stopwords.txt")]

        sampledTrainingData = trainingData.sample(frac=0.5)
        numWords, vectorizer, theta0, theta1, theta_j_0, theta_j_1 = train(sampledTrainingData)
        for x in range(10):
            # TODO modify the following line to test actual test data
            # noinspection PyRedeclaration
            sampledTestData = trainingData.sample(frac=0.5)

            # Test and get predictions
            test(sampledTestData, theta0, theta1, theta_j_0, theta_j_1)
    else:
        print("Colab environment not detected. Running program in low RAM usage mode\n")
        trainingData = pd.read_csv("train.csv")  # | review (text)  | sentiment (pos/neg) |
        # testData = pd.read_csv("test.csv")       # | id (review id) |   review (text)     |
        stopWords = [line.rstrip('\n') for line in open("stopwords.txt")]

        sampledTrainingData = trainingData.sample(frac=0.1)

        # Train and get the required values for our thetas
        numWords, vectorizer, theta0, theta1, theta_j_0, theta_j_1 = train(sampledTrainingData)

        # TODO modify the following line to test actual test data
        # noinspection PyRedeclaration
        sampledTestData = trainingData.sample(frac=0.1)

        # Test and get predictions
        test(sampledTestData, theta0, theta1, theta_j_0, theta_j_1)


