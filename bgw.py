import nltk
import pandas as pd 
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn import metrics
import numpy as np
from sklearn import cross_validation
from nltk.corpus import stopwords

PATH_TO_TRAIN_DATA_1 = '/Users/hewang/Desktop/Train and Test/6000_Test/sarc_train_13326.txt'
PATH_TO_TRAIN_DATA_2 = '/Users/hewang/Desktop/Train and Test/6000_Test/norm_train_13326.txt'

PATH_TO_TEST_DATA_1 = '/Users/hewang/Desktop/Train and Test/6000_Test/sarc_test_6000.txt'
PATH_TO_TEST_DATA_2 = '/Users/hewang/Desktop/Train and Test/6000_Test/norm_test_6000.txt'


def review_to_words( raw_review ):
    # Remove HTML
    review_text = BeautifulSoup(raw_review,"lxml").get_text() 
    # Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    # Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    # Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    # Join the words back into one string separated by space, and return the result.
    return( " ".join( meaningful_words ))   

def make_dateset(srac_read, norm_read):
    tweets = []
    sarcasm_label = []
    fh = open(srac_read)
    train_lines = fh.readlines()
    fh.close()
    for line in train_lines:
        line = review_to_words(line)
        tweets.append(line)
        sarcasm_label.append(1)

    fh2 = open(norm_read)
    train_lines2 = fh2.readlines()
    fh2.close()
    for line in train_lines2:
        line = review_to_words(line)
        tweets.append(line)
        sarcasm_label.append(0)

    return tweets, sarcasm_label

train_dataset, train_labels = make_dateset(PATH_TO_TRAIN_DATA_1, PATH_TO_TRAIN_DATA_2)
test_dataset, test_labels = make_dateset(PATH_TO_TEST_DATA_1, PATH_TO_TEST_DATA_2)


def tokens(x):
    return x.split(',')
stopWords = stopwords.words('english')

count_vectorizer = CountVectorizer(analyzer = "word",   \
                                   tokenizer = None,    \
                                   preprocessor = None, \
                                   stop_words = stopWords,
                                   ngram_range=(2,3))


tfidf_transformer = TfidfTransformer(use_idf=True)

# bag of words
train_data_features_1 = count_vectorizer.fit_transform(train_dataset)
# tf-idf
train_data_features_2 = tfidf_transformer.fit_transform(train_data_features_1)

test_data_features_1 = count_vectorizer.fit_transform(test_dataset)
test_data_features_2 = tfidf_transformer.fit_transform(test_data_features_1)

#print train_data_features_1.shape

# final train dataset
#nmf_train = NMF(n_components=10, random_state=1, alpha=.1, l1_ratio=.5).fit_transform(train_data_features_1)
#nmf_test = NMF(n_components=10, random_state=1, alpha=.1, l1_ratio=.5).fit_transform(test_data_features_1)
"""
lda_train = LatentDirichletAllocation(n_topics=12, max_iter=5,
                                      learning_method='online', learning_offset=50.,
                                      random_state=0).fit_transform(train_data_features_1)
lda_test = LatentDirichletAllocation(n_topics=12, max_iter=5,
                                      learning_method='online', learning_offset=50.,
                                      random_state=0).fit_transform(test_data_features_1)
"""
#lda.fit_transform(train_data_features_1)
#print lda.shape

#print nmf_2.shape
#print lda.shape
train_data_features = train_data_features_2.toarray()
test_data_features = train_data_features_2.toarray()

#print train_data_features.shape

#vocab = vectorizer.get_feature_names()
#print vocab

# Random forest tree
# Initialize a Random Forest classifier with 100 trees

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
"""
def fillAlist(n):
    newlist=[]
    for i in range(0,n):
        newlist.append(1)
    return newlist

train_labels = fillAlist(7329)
#test_labels = fillAlist(7329)
"""

"""
# random forest
forest = RandomForestClassifier(n_estimators = 100) 
forest = forest.fit(nmf_train, train_labels)
predicted = forest.predict(nmf_test)
"""


# GaussianNB
gnb = GaussianNB()
predicted = gnb.fit(train_data_features, train_labels).predict(test_data_features)

# cross-validation
print '5-fold cross_validation score:'
scores = cross_validation.cross_val_score(gnb, train_data_features, train_labels, cv=5)
print scores
# prediction accuracy
print metrics.classification_report(test_labels, predicted, target_names = [str(0),str(1)])
print metrics.confusion_matrix(test_labels, predicted)
print np.mean(predicted == test_labels)
"""
"""
"""
# LinearSVC
lsvc = LinearSVC()
predicted = lsvc.fit(lda_train, train_labels).predict(lda_test)

# cross-validation
print '5-fold cross_validation score:'
scores = cross_validation.cross_val_score(lsvc, lda_train, train_labels, cv=5)
print scores
# prediction accuracy
print metrics.classification_report(test_labels, predicted, target_names = [str(0),str(1)])
print metrics.confusion_matrix(test_labels, predicted)
print np.mean(predicted == test_labels)
"""

"""
# hyper-parameter selection for NMF

list_n_components = [10, 50, 100, 250, 500]
for x in list_n_components:
    nmf_train = NMF(n_components=x, random_state=1, alpha=.1, l1_ratio=.5).fit_transform(train_data_features_1)
    nmf_test = NMF(n_components=x, random_state=1, alpha=.1, l1_ratio=.5).fit_transform(test_data_features_1)

    gnb = GaussianNB()
    predicted = gnb.fit(nmf_train, train_labels).predict(nmf_test)

    scores = cross_validation.cross_val_score(gnb, nmf_train, train_labels, cv=5)

    print 'number of n_components using now %d' %x
    print '5-fold cross_validation score:'
    print scores
    # prediction accuracy
#    forest = forest.fit(nmf_train, train_labels)
#    predicted = forest.predict(nmf_test)

    print metrics.classification_report(test_labels, predicted, target_names = [str(0),str(1)])
    print metrics.confusion_matrix(test_labels, predicted)
    print np.mean(predicted == test_labels)
"""
"""
# hyper-parameter selection for LDA
#list_n_topics = [10, 50, 100, 250, 500]
list_n_topics = [11,12,13,14,15]
for x in list_n_topics:
    lda_train = LatentDirichletAllocation(n_topics=x, max_iter=5,
                                      learning_method='online', learning_offset=50.,
                                      random_state=0).fit_transform(train_data_features_1)
    lda_test = LatentDirichletAllocation(n_topics=x, max_iter=5,
                                      learning_method='online', learning_offset=50.,
                                      random_state=0).fit_transform(test_data_features_1)
    gnb = GaussianNB()
    predicted = gnb.fit(lda_train, train_labels).predict(lda_test)

    scores = cross_validation.cross_val_score(gnb, lda_train, train_labels, cv=5)
    print "number of n_topics using now %d" % x
    print '5-fold cross_validation score:'
    print scores

    print metrics.classification_report(test_labels, predicted, target_names = [str(0),str(1)])
    print metrics.confusion_matrix(test_labels, predicted)
    print np.mean(predicted == test_labels)

"""
"""
# hyper-parameter selection for random forest tree
#list_n_estimators = [10, 25, 50, 75, 100, 125, 150, 200, 500]
list_n_estimators = range(5,16)
for x in list_n_estimators:
    forest = RandomForestClassifier(n_estimators = x)
    forest = forest.fit(lda_train, train_labels)
    predicted = forest.predict(lda_test) 

    print "number of n_estimators using now %d" % x
    scores = cross_validation.cross_val_score(forest, lda_train, train_labels, cv=5)
    print '5-fold cross_validation score:'
    print scores

    print metrics.classification_report(test_labels, predicted, target_names = [str(0),str(1)])
    print metrics.confusion_matrix(test_labels, predicted)
    print np.mean(predicted == test_labels)
"""

