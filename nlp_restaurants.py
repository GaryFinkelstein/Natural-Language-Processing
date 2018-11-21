import pandas as pd
import numpy as np

df = pd.read_csv("Restaurant_Reviews.tsv", delimiter= "\t", quoting=3)

#=====================
#Part 1. Cleaning the data
#=====================

#Removing punctuation and stop words (very common words):
import string
from nltk.corpus import stopwords 

stop_words = stopwords.words("English") #very common words in english

def text_cleaner(message):
    no_punc_mess = [letter for letter in message if letter not in string.punctuation] #removing punctuation from review message and storing letters of message in a list
    no_punc_mess = "".join(no_punc_mess).split() #joining the letters to reform the message without punctuation and is now a string again
    clean_message_list  = [word for word in no_punc_mess if word.lower() not in stopwords.words("english")] #removing stop words and returning clean tokenised message
    
    return clean_message_list



#========================================
#Part 2. Bagging - vectorising the tokenised data
#(Can ignore Part 2 and skip to part 3 as this section is designed to help understand the inner-workings of the bagging process
#========================================

from sklearn.feature_extraction.text import CountVectorizer

bag_of_words_transformer = CountVectorizer(analyzer=text_cleaner) #we specify the analyzer to be our own text_cleaner function above to tell the bow_transformer which features to vectorise
bag_of_words_transformer.fit(df.Review) #fitting to the data (have not yet transfomerd the data into vectorised count form)

#------
# insights into the bag_of_words_transformer
print("Vocabulary list:")
print("")
print(bag_of_words_transformer.vocabulary_)
print("")
print("Extracting a word from the vocabulary list, index number 463:", bag_of_words_transformer.get_feature_names()[463])
print("")
example_review = df.Review[7]
print("Example of what the bag_of_words_transformer is doing: \n")
print("Review:" , example_review)
print("")
print("bag of words transformed review:" )
print(bag_of_words_transformer.transform([example_review])) #vector of counts (the number of times each word appears)
print("")
#--------

review_bow = bag_of_words_transformer.transform(df.Review) #each message now described by a vector of counts
print("Shape of sparse review matrix", review_bow.shape) #num messages x num words in total 
print("")
print("Bow transformed reviews - reviews as vector of counts:")
print("")
print(review_bow)
print("")


#Tfidf re-weighting of words to give more weight to those words that are less common per message and appear in less messages over all reviews
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
tfidf_reviews = tfidf_transformer.fit_transform(review_bow)
print("Tfidf transformerd reviews:")
print("")
print(tfidf_reviews)
print("")

#============================================================================================
#Part 3. Implementing a machine learning classifier to classify the messages as positive or negative
#============================================================================================
#In order to avoid having to do the above steps for a training set, sklearns pipeline is used to create an ordered operation on the reviews data set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.Review, df.Liked, test_size=0.3, random_state=101)

from sklearn.feature_extraction.text import CountVectorizer #get word counts per message matrix
from sklearn.feature_extraction.text import TfidfTransformer #get tfidf values for each word
from sklearn.naive_bayes import MultinomialNB #classifier

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("bow", CountVectorizer(analyzer=text_cleaner)) , ("tfidf", TfidfTransformer()) , ("Classifier", MultinomialNB())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))