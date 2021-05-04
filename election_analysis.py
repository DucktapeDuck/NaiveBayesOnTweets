# Sentiment of tweets before and after the election.
import pandas as pd
import datetime
import numpy as np

politics = pd.read_csv('./data/output.csv')
politics_data = politics.set_index('idx').T.to_dict()

# Election results started
election_start = datetime.datetime.strptime('2020-11-03 18:00:00', '%Y-%m-%d %H:%M:%S')
# First Official Call by Decision Desk (Most were called on Nov 7th tho.
election_called = datetime.datetime.strptime('2020-11-07 08:50:00', '%Y-%m-%d %H:%M:%S')

before_election = []
during_election = []
after_election = []

for tweet in politics_data:
    time = politics_data[tweet]['Time'][0:-6]
    time = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')

    if time < election_start:
        before_election.append(int(politics_data[tweet]['preds']))
    elif time < election_called:
        during_election.append(int(politics_data[tweet]['preds']))
    else:
        after_election.append(int(politics_data[tweet]['preds']))

print("Total Tweets:")
print('Before Election:', len(before_election))
print('During Election:', len(during_election))
print('After Election:', len(after_election))

print("\nElection Sentiment Averages:")
print('Before Election:', np.sum(before_election) / len(before_election))
print('During Election:', np.sum(during_election) / len(during_election))
print('After Election:', np.sum(after_election) / len(after_election))

