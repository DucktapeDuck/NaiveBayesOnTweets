from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

test = pd.read_csv('./data/t_test.txt', sep='\t')
train = pd.read_csv('./data/t_train.txt', sep='\t')

doc = {}
doc[0] = train[train.iloc[:, 0] == 0].iloc[:, 1].to_numpy()
# print(doc[0])
doc[1] = train[train.iloc[:, 0] == 1].iloc[:, 1].to_numpy()

# print(train.to_numpy())
# print(train.iloc[:,0])
#
# print(train[train.iloc[:,0] == '0'].iloc[:,1].to_numpy())
# print(doc)
# print(list(filter(lambda i: i==0, )))

def extract_vocab(tweets):
    vocab = set()

    for tweet in tweets:
        print(tweet)
        for word in tweet.split(' '):
            vocab.add(word.lower())

    return vocab


def count_words(doc):
    counts = {}

    for classes in [0, 1]:
        class_tweets = doc[classes]
        counts[classes] = defaultdict(int)
        for tweet in list(class_tweets):
            # print(tweet)
            words = tweet.split(' ')
            for word in words:
                counts[classes][word.lower()] += 1

    return counts

def predict(tweet, prior, likelihoods, vocab):
    sums = {
        0: 0,
        1: 0,
    }
    for classes in [0,1]:
        sums[classes] = prior[classes]
        # print(tweet)
        words = tweet.split(' ')
        for word in words:
            if word in vocab:
                sums[classes] += likelihoods[classes][word.lower()]
    
    if sums[0] > sums[1]:
        return 0
    else:
        return 1



training_data = list(train.iloc[:, 1])
training_labels = list(train.iloc[:, 0])
testing_data = list(test.iloc[:, 1])
testing_labels = list(test.iloc[:, 0])

vocab = extract_vocab(training_data)

N_doc = len(training_data)
counts = count_words(doc)

# print(counts)
logprior = {}
loglikelihoods = { 0: {}, 1: {},}


# print(sum(training_labels == '0'))
for classes in [0, 1]:
    N_c = 0.0
    for item in training_labels:
        if int(item) == classes:
            N_c += 1.
    print(str(classes) + ' ' + str(N_c))

    logprior[classes] = np.log(N_c / N_doc)

    total_count = 0
    for word in vocab:
        total_count += counts[classes][word]
    # print(total_count)

    for word in vocab:
        count = counts[classes][word]
        loglikelihoods[classes][word] = np.log((count + 1.0) / (total_count + 1.0 + len(vocab)))

# print(predict('I HATE THIS', logprior, loglikelihoods, vocab))
testing_preds = [predict(tweet.lower(), logprior, loglikelihoods, vocab) for tweet in testing_data]

print(accuracy_score(testing_labels, testing_preds))

