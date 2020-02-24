
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import sys
from sklearn.model_selection import KFold
import nltk
nltk.download('stopwords')

maxFeatures = 5000
inColab = False  # No need to change this, it's checked automatically

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

    tempDict['positive'] = np.array(tempDict['positive'], dtype=np.uint32)
    tempDict['negative'] = np.array(tempDict['negative'], dtype=np.uint32)
    return tempDict

def train(trainingData, stopWords):

    # 0 - 84.38% (50000 words, 5 splits)
    # vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True, analyzer='word', max_features=maxFeatures, binary=True)

    # 1 - 84.25% (50000 words, 5 splits)
    # vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True, analyzer='word', token_pattern='(?u)[a-zA-Z]+', max_features=maxFeatures, binary=True)

    # 2 - 85.06% (50000 words, 5 splits)
    # vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True, analyzer='word', token_pattern='(?u)[a-zA-Z]+', stop_words=stopWords, max_features=maxFeatures, binary=True)

    # 3 - 85.27% (50000 words, 5 splits)
    vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True, analyzer='word', token_pattern='(?u)[a-zA-Z]+|\\b\\w\\/\\w+\\b', stop_words=stopWords, max_features=maxFeatures, binary=False)

    countMatrix = vectorizer.fit_transform(trainingData['review'])
    countMatrixAsArray = countMatrix.toarray()
    # print("Show the", maxFeatures, "most popular words in the corpus:\n", vectorizer.get_feature_names())
    # print("\nNumber of reviews (rows) in the word count table:", len(countMatrixAsArray))
    # print("Number of words   (cols) in the word count table:", len(countMatrixAsArray[0]))
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
    theta_j_0 = np.array([0 for _ in range(numWords)], dtype='float64')
    theta_j_1 = np.array([0 for _ in range(numWords)], dtype='float64')

    for x in range(numWords):
        theta_j_0[x] = (negCounts[x] + 1) / (totalNegReviews + numWords)
        theta_j_1[x] = (posCounts[x] + 1) / (totalPosReviews + numWords)

    return numWords, vectorizer, theta0, theta1, theta_j_0, theta_j_1


def test(vectorizer, testData, theta0, theta1, theta_j_0, theta_j_1):
    transformedData = vectorizer.transform(testData['review'])
    dataToArray = transformedData.toarray()
    labelVector = testData['sentiment'].to_numpy()

    # This complicated looking mess is simply a log-likelihood calculation that's been compressed to be memory efficient
    # It was originally done using multiple variables but this proved to be problematic because of very high memory usage
    fastLogLikelihoods = np.add(np.add(np.dot(dataToArray, np.log(np.true_divide(theta_j_1, theta_j_0))), np.dot(
        np.subtract(np.ones((len(dataToArray), len(dataToArray[0]))), dataToArray), np.log(
            np.true_divide(np.subtract(np.ones(len(theta_j_1)), theta_j_1),
                           np.subtract(np.ones(len(theta_j_0)), theta_j_0))))), np.log(theta1 / theta0))

    correctPredictions = 0
    for i in range(len(testData)):
        if fastLogLikelihoods[i] > 0 and labelVector[i] == 'positive':
            correctPredictions += 1
        if fastLogLikelihoods[i] < 0 and labelVector[i] == 'negative':
            correctPredictions += 1
    return 100 * correctPredictions / len(labelVector)


if __name__ == "__main__":
    inColab = 'google.colab' in sys.modules
    if inColab:
        print("Colab environment detected.\n")
        maxFeatures = 50000
        trainingData = pd.read_csv("./gdrive/My Drive/train.csv")  # | review (text)  | sentiment (pos/neg) |
        # testData = pd.read_csv("test.csv")       # | id (review id) |   review (text)     |
        stopWords = [line.rstrip('\n') for line in open("./gdrive/My Drive/stopwords.txt")]

        trainingData = trainingData.sample(frac=1)

        kf = KFold(n_splits=5, shuffle=False)
        foldNum = 0
        foldAccs = []
        for train_index, test_index in kf.split(trainingData):
            print("Fold", foldNum + 1)

            print("Training...", end='')
            kfoldTrain = trainingData.iloc[train_index]
            numWords, vectorizer, theta0, theta1, theta_j_0, theta_j_1 = train(kfoldTrain, stopWords)
            print("DONE")

            print("Testing...", end='')
            kfoldTest = trainingData.iloc[test_index]
            acc = test(vectorizer, kfoldTest, theta0, theta1, theta_j_0, theta_j_1)
            print("DONE \nAccuracy = {0:.2f}".format(acc), "%\n")

            foldAccs.append(acc)
            foldNum += 1
        print("Average accuracy across all folds = {0:.2f}%".format(np.array(foldAccs).sum() / len(foldAccs)))



    else:
        print("Colab environment not detected.\n")
        trainingData = pd.read_csv("train.csv")  # | review (text)  | sentiment (pos/neg) |
        # testData = pd.read_csv("test.csv")       # | id (review id) |   review (text)     |
        stopWords = [line.rstrip('\n') for line in open("stopwords.txt")]

        trainingData = trainingData.sample(frac=0.5)

        kf = KFold(n_splits=10, shuffle=False)
        foldNum = 0
        foldAccs = []
        for train_index, test_index in kf.split(trainingData):
            print("Fold", foldNum + 1)

            print("Training...", end='')
            kfoldTrain = trainingData.iloc[train_index]
            numWords, vectorizer, theta0, theta1, theta_j_0, theta_j_1 = train(kfoldTrain, stopWords)
            print("DONE")

            print("Testing...", end='')
            kfoldTest = trainingData.iloc[test_index]
            acc = test(vectorizer, kfoldTest, theta0, theta1, theta_j_0, theta_j_1)
            print("DONE \nAccuracy = {0:.2f}".format(acc), "%\n")

            foldAccs.append(acc)
            foldNum += 1
        print("Average accuracy across all folds = {0:.2f}%".format(np.array(foldAccs).sum() / len(foldAccs)))
