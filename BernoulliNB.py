import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import sys
from sklearn.model_selection import KFold
import time

maxFeatures = 5000  # Maximum number of words to use with the vectorizer
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


# Train the algorithm. Produces the vectorizer and thetas used for classification
def train(trainingData, stopWords):
    # Create the vectorizer and fit it to the data. Transform the training data to word counts.
    vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True, analyzer='word', token_pattern='(?u)[a-zA-Z]+',
                                 stop_words=stopWords, max_features=maxFeatures, binary=True)
    vectorizer = CountVectorizer(strip_accents='ascii', max_features=maxFeatures ,binary=True)
    countMatrix = vectorizer.fit_transform(trainingData['review'])
    countMatrixAsArray = countMatrix.toarray()  # This line gets us the training examples as a matrix, where rows are reviews and columns are word counts
    numWords = len(countMatrixAsArray[0])

    # ----- Uncomment the following lines to see some info printed out during the training process -----
    # print("Show the", maxFeatures, "most popular words in the corpus:\n", vectorizer.get_feature_names())
    # print("\nNumber of reviews (rows) in the word count table:", len(countMatrixAsArray))
    # print("Number of words   (cols) in the word count table:", len(countMatrixAsArray[0]))

    # Create dictionary with separated classes
    separated = separateClasses(countMatrixAsArray, trainingData['sentiment'])

    # Calculate theta0 and theta1
    totalNegReviews = len(separated['negative'])
    totalPosReviews = len(separated['positive'])
    totalReviews = totalNegReviews + totalPosReviews
    theta0 = totalNegReviews / totalReviews
    theta1 = totalPosReviews / totalReviews

    # Calculate theta_j_0 and theta_j_1
    negCounts = separated['negative'].sum(axis=0)
    posCounts = separated['positive'].sum(axis=0)
    theta_j_0 = np.array([0 for _ in range(numWords)], dtype='float64')
    theta_j_1 = np.array([0 for _ in range(numWords)], dtype='float64')
    for x in range(numWords):
        theta_j_0[x] = (negCounts[x] + 1) / (totalNegReviews + numWords)
        theta_j_1[x] = (posCounts[x] + 1) / (totalPosReviews + numWords)

    # Return all we'll need for testing
    return numWords, vectorizer, theta0, theta1, theta_j_0, theta_j_1


# Test how well our algorithm can classify new examples
def test(vectorizer, testData, theta0, theta1, theta_j_0, theta_j_1):
    # Convert the reviews from words to to word counts in accordance with the vectorizer we used in training
    transformedData = vectorizer.transform(testData['review'])
    dataToArray = transformedData.toarray()

    # For convenience and clarity, assign the sentiment column to "labelVector"
    labelVector = testData['sentiment'].to_numpy()

    # This complicated looking mess is simply a log-likelihood calculation that's been compressed to be memory efficient
    # It was originally done using multiple variables but this proved to be problematic because of very high memory usage
    fastLogLikelihoods = np.add(np.add(np.dot(dataToArray, np.log(np.true_divide(theta_j_1, theta_j_0))), np.dot(
        np.subtract(np.ones((len(dataToArray), len(dataToArray[0]))), dataToArray), np.log(
            np.true_divide(np.subtract(np.ones(len(theta_j_1)), theta_j_1),
                           np.subtract(np.ones(len(theta_j_0)), theta_j_0))))), np.log(theta1 / theta0))

    # Loop over our prediction vector and count up how many correct predictions were made
    correctPredictions = 0
    for i in range(len(testData)):
        if fastLogLikelihoods[i] > 0 and labelVector[i] == 'positive':
            correctPredictions += 1
        if fastLogLikelihoods[i] < 0 and labelVector[i] == 'negative':
            correctPredictions += 1

    # Return an accuracy score out of 100
    return 100 * correctPredictions / len(labelVector)


if __name__ == "__main__":

    # To allow for easier cross platform execution, check if this file is being run in a Google Colab notebook
    inColab = 'google.colab' in sys.modules

    # If we're using Colab, then we can work with bigger amounts of data
    if inColab:
        print("Colab environment detected.\n")
        trainingData = pd.read_csv("./gdrive/My Drive/train.csv")  # | review (text)  | sentiment (pos/neg) |
        # testData = pd.read_csv("test.csv")       # | id (review id) |   review (text)     |
        stopWords = [line.rstrip('\n') for line in open("./gdrive/My Drive/stopwords.txt")]

        # Take n random samples, where n = (number of rows in dataset) * frac
        # Setting frac=1 is the same as shuffling the rows of the dataset
        trainingData = trainingData.sample(frac=1)

        # Do K-Fold cross-validation. Call our train/test functions
        # Print the accuracy of each fold and the average accuracy across all folds
        kf = KFold(n_splits=5, shuffle=False)
        foldNum = 0
        foldAccs = []
        startTime = time.time()
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
        print("Total runtime = {0:.2f} seconds".format(time.time() - startTime))
        print("Average run time per fold = {0:.2f} seconds".format((time.time() - startTime)/foldNum))


    # If we're not using Colab, then we'll work with more modest amounts of data (between 1/4 to 1/2 or so)
    else:
        print("Colab environment not detected.\n")
        trainingData = pd.read_csv("train.csv")  # | review (text)  | sentiment (pos/neg) |
        # testData = pd.read_csv("test.csv")       # | id (review id) |   review (text)     |
        stopWords = [line.rstrip('\n') for line in open("stopwords.txt")]

        # Take n random samples, where n = (number of rows in dataset) * frac
        # Setting frac=0.5 means we're randomly picking half of the rows in the dataset
        trainingData = trainingData.sample(frac=0.5)

        # Do K-Fold cross-validation. Call our train/test functions
        # Print the accuracy of each fold and the average accuracy across all folds
        kf = KFold(n_splits=5, shuffle=False)
        foldNum = 0
        foldAccs = []
        startTime = time.time()
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
        print("Total runtime = {0:.2f} seconds".format(time.time() - startTime))
        print("Average run time per fold = {0:.2f} seconds".format((time.time() - startTime)/foldNum))
