
# TODO
#  maybe use TF-IDF features with sklearn (TfidfVectorizer or TfidfTransformer)
#  https://towardsdatascience.com/hacking-scikit-learns-vectorizers-9ef26a7170af   might have ways of improving results

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from random import shuffle

maxFeatures = 5000

# CURRENTLY NOT USED - this function was from previous stuff I was testing. Might come in handy later
# Split data up, return a dictionary where keys are class names and values are lists of entries
def separateClasses(data):
    tempDict = dict()
    tempDict['positive'] = list()
    tempDict['negative'] = list()
    for x in range(len(data)):
        if data[x][1] == 'positive':
            tempDict['positive'].append(data[x][0])
        else:
            tempDict['negative'].append(data[x][0])
    return tempDict

if __name__ == "__main__":
    trainingData = pd.read_csv("train.csv")  # | review (text)  | sentiment (pos/neg) |
    testData = pd.read_csv("test.csv")       # | id (review id) |   review (text)     |
    stopWords = [line.rstrip('\n') for line in open('stopwords.txt')]

    # Shuffle the rows for better accuracy
    trainingData = trainingData.sample(frac=1)

    # Only work on a quarter of the reviews while we're building/debugging (for speed)
    trainingData = trainingData.iloc[:len(trainingData)//4]

    vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True, analyzer='word', token_pattern='[a-zA-Z]+', stop_words=stopWords, max_features=maxFeatures, binary=False)
    countMatrix = vectorizer.fit_transform(trainingData['review'])
    array = countMatrix.toarray()
    print("Show the", maxFeatures, "most popular words in the corpus\n", vectorizer.get_feature_names())
    print("\nNumber of reviews (rows) in the 'dictionary':", len(array))
    print("Number of words   (cols) in the 'dictionary':", len(array[0]))
    print("\nThe 'dictionary' (aka word count table):\n", array)



