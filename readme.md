# Naive Bayes for Tweet Sentiment Classification

These python programs attempt to learn the sentiment of tweets and predict the sentiments of tweets regarding the election.

## Installation

Use the package manager anaconda to install the following packages:

* Numpy
* Pandas
* Sklearn

## Usage

The following commands run the programs given the 

```bash
python ./naive_bayes_classifier.py 
python ./naive_bayes_classifier_analysis.py 
python ./election_analysis.py
```

For naive_bayes_classifier.py, this file outputs the election predictions (election_output.csv is the latest calculated output). This also ingests data from the ./data directory. 

For naive_bayes_classifier.py, this file outputs the graph seen in the directory ./plots/.

The election_analysis.py provides the time analysis regarding the election. 

It is assumed the path of python is the base of this directory for all python files, as to keep the integrity of all reads of the ./data/ files.   

## Contributing

Author: Connor Frost (Developer)
