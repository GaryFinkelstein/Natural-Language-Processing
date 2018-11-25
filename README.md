# Natural-Language-Processing
Applying Natural Language Processing to a restaurant reviews dataset to predict whether or not a review of the restaurant was positive or negative.

#
#### Python packages used:
- Pandas
- Numpy
- String
- Nltk
- Sklearn

#
#### Process of performing natural language processing on restaurant reviews dataset:

- Step 1: Cleaning data - removing punctuation and commonly used words from the review messages.
- Step 2: Performing a Train-Test split on the observed data
- Step 3: Bagging - transforming the review messages to a vector of word counts. 
- Step 4: Tfidf step - re-weighting word counts in a vector to give more weight to those words that are less common per message and appear in less messages over all reviews
- Step 5: Classification - applying a classification algorithmm, in this case Mulitnomial Naive Bayes, to classify the reviews as positive or negative 

#
#### The model results:

- Achieves an F1 score of 75%
