# Connor Frost (FrostC)
# Reference: https://medium.datadriveninvestor.com/implementing-naive-bayes-for-sentiment-analysis-in-python-951fa8dcd928?gi=5a71d9a48a05

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

test = pd.read_csv('./data/t_test.txt', sep='\t', header=None)
train = pd.read_csv('./data/t_train.txt', sep='\t', header=None)

doc = {}
doc[0] = train[train.iloc[:, 0] == 0].iloc[:, 1].to_numpy()
doc[1] = train[train.iloc[:, 0] == 1].iloc[:, 1].to_numpy()

# Words to skip training on.
def skip_word(word):
    if len(word) == 0:
        return True

    if word in ['a', 'an', 'the', 'and', 'for']:
        return True

    if word[0] == '@':
        return True

    if 'http' in word:
        return True

    return False


# Extract the words from each and every tweet to add to the vocab with the exlusion of skipped words
def extract_vocab(tweets):
    vocab = set()

    for tweet in tweets:
        for word in tweet.split(' '):
            if skip_word(word.lower()):
                continue
            vocab.add(word.lower())

    return vocab

# Count the words and their associated classification.
def count_words(tweets):
    counts = {}

    for classes in [0, 1]:
        # get the tweets corresponding to each class
        class_tweets = tweets[classes]

        # Create a list of counts to add
        counts[classes] = defaultdict(int)

        # Count the words with the exception of excluded words
        for tweet in list(class_tweets):
            words = tweet.split(' ')
            for word in words:
                if skip_word(word.lower()):
                    continue
                counts[classes][word.lower()] += 1

    # Return computed counts
    return counts

def predict(tweet, prior, likelihoods, vocab):
    # Create a dictionary to add the sentiments to.
    sums = {
        0: 0,
        1: 0,
    }

    # Calculate the projected weights of the sentiments using the prior and likelihoods
    for classes in [0,1]:
        sums[classes] = prior[classes]
        words = tweet.split(' ')
        for word in words:
            # If the word is in the vocab, then return
            if word in vocab:
                sums[classes] += likelihoods[classes][word.lower()]

    # Generate the prediction here and return the corresponding prediction
    if sums[0] > sums[1]:
        return 0
    else:
        return 1

# Create the data
training_data = list(train.iloc[:, 1])
training_labels = list(train.iloc[:, 0])
testing_data = list(test.iloc[:, 1])
testing_labels = list(test.iloc[:, 0])

# For collecting the accuracies
y_test = []
y_train = []

for epochs in range(100, 12000, 100):
    # Call the above functions
    vocab = extract_vocab(training_data[:epochs])
    N_doc = len(training_data[:epochs])
    counts = count_words(doc)

    # Create the priors and likelihoods
    logprior = {}
    loglikelihoods = {0: {}, 1: {}, }

    for classes in [0, 1]:
        # Count the tweets in each class
        N_c = 0.0
        for item in training_labels:
            if int(item) == classes:
                N_c += 1.
        print(str(classes) + ' ' + str(N_c))

        # Compute the logpriors
        logprior[classes] = np.log(N_c / N_doc)

        # Compute the total number of counts
        total_count = 0
        for word in vocab:
            total_count += counts[classes][word]

        # Compute the likelihoods
        for word in vocab:
            count = counts[classes][word]
            loglikelihoods[classes][word] = np.log((count + 1.0) / (total_count + 1.0 + len(vocab)))

    # Compute the accuracy at this step and append it to the list
    print("ACCURACY |", str(epochs))
    testing_preds = [predict(tweet.lower(), logprior, loglikelihoods, vocab) for tweet in testing_data]
    y_test.append(accuracy_score(testing_labels, testing_preds))
    training_preds = [predict(tweet.lower(), logprior, loglikelihoods, vocab) for tweet in training_data]
    y_train.append(accuracy_score(training_labels, training_preds))

# Create the plot
x = range(100, 12000, 100)
plt.plot(x, y_test, color='red', label='Test')
plt.plot(x, y_train, color='blue', label='Train')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('./plots/Epoch_Accuracy.png')

# Print accuracies for viewing
print(y_test)
print(y_train)



