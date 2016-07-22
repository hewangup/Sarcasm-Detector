import json
import sklearn
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import random

def train_test_split(sarc_read, norm_read, test_size):
    probability = 0.3
    train_count = 0
    test_count = 0
    count = 0
    writeMe_train = open('sarc_train.txt', 'w')
    writeMe_test = open('sarc_test.txt','w')
    with open(sarc_read) as readMe:
        for line in readMe:
            tweet = str(json.loads(line))
            if(count < test_size):
                if(random.random()>probability):
                    count += 1				
                    test_count += 1
                    writeMe_test.write(json.JSONEncoder().encode(tweet)+"\n")
                else:
                    writeMe_train.write(json.JSONEncoder().encode(tweet)+"\n")
                    train_count += 1
            else:
                writeMe_train.write(json.JSONEncoder().encode(tweet)+"\n")
                train_count += 1

    writeMe_train = open('norm_train.txt', 'w')
    writeMe_test = open('norm_test.txt', 'w')
    count = 0
    with open(norm_read) as readMe:
        for line in readMe:
            tweet = str(json.loads(line))
            if(count < test_size):
                if(random.random()>probability):
                    count += 1				
                    test_count += 1
                    writeMe_test.write(json.JSONEncoder().encode(tweet)+"\n")
                else:
                    writeMe_train.write(json.JSONEncoder().encode(tweet)+"\n")
                    train_count += 1
            else:
                writeMe_train.write(json.JSONEncoder().encode(tweet)+"\n")
                train_count += 1

    print('Test: ' + str(test_count) + " Train: " + str(train_count))

#train_test_split('compiled_tweets_sarcasm.txt', 'compiled_tweets_lol.txt', 2500)

#Makes the Test Set
#Sarc_read = Test set file of sarcastic tweets
#Norm_read = Test set file of normal tweets
def make_Test_Set(sarc_read, norm_read):
    tweets = []
    sarcasm_label = []
    with open(sarc_read) as readMe:
        for line in readMe:
            tweet = json.loads(line)
            tweets.append(str(tweet))
            sarcasm_label.append(1)
    with open(norm_read) as readMe:
        for line in readMe:
            tweet = json.loads(line)
            tweets.append(str(tweet))
            sarcasm_label.append(0)
    return (tweets, sarcasm_label)


#Takes a training set file of sarcastic tweets and returns the cluster assignment of each tweet	
def cluster_assignment_sarc(tweets, k):
    features = 1000
    tfidf = sklearn.feature_extraction.text.TfidfVectorizer(decode_error='ignore', stop_words=None, lowercase=True, ngram_range=(2,3), max_features = features)
    tweets_tfidf = tfidf.fit_transform(tweets)
    k_fit = KMeans(n_clusters = k).fit(tweets_tfidf)
    return (k_fit, k_fit.labels_, tfidf)

#Takes a file of tweets and returns them as a list	
def tweet_list(tweet_file):
    tweets = []
    with open(tweet_file) as readMe:
        for line in readMe:
           tweet = json.loads(line)
           tweets.append(str(tweet))
    return tweets

#Performs prediction and prints out performance metrics
from sklearn import metrics
def prediction_tfidf(count_vect, tfidf, mNB, tweets, labels):
    train_pred = count_vect.transform(tweets)
    train_predi = tfidf.transform(train_pred)
    predicted = mNB.predict(train_predi)
    print(metrics.classification_report(labels, predicted, target_names = [str(0),str(1)]))
    print(metrics.confusion_matrix(labels, predicted))
    print("\n")
    return np.mean(predicted == labels)

def prediction_cv(count_vect, mNB, tweets, labels):
    train_pred = count_vect.transform(tweets)
    predicted = mNB.predict(train_pred)
    print(metrics.classification_report(labels, predicted, target_names = [str(0),str(1)]))
    print(metrics.confusion_matrix(labels, predicted))
    print("\n")
    return np.mean(predicted == labels)

#Create k models using the cluster assignments of each sarcastic tweet.
#Sarc_tweets_file = File of sarcastic tweets used for TRAINING SET
#Norm_tweets_file = File of normal tweets used for TRAINING SET
#K = The number of models and number of clusters to make
#Test_Tweets = List of Test Tweets
#Test_Labels = List of Test Labels
def k_models_classifier(sarc_tweets_file, norm_tweets_file, k, test_tweets, test_labels):

    sarc_tweets_train = tweet_list(sarc_tweets_file)
    norm_tweets_train = tweet_list(norm_tweets_file)
    c = cluster_assignment_sarc(sarc_tweets_train, k)
    cluster_assignments = c[1]
    for i in range(0, 100):
        print(cluster_assignments[i])
    k_fit = c[0]
    tfidf_vect = c[2]

    models = []
    probability = 0.5					#This might be a value that you want to experiment with
    for i in range(0, k):
        sarc_count = 0	
        train_tweets = []
        train_labels = []
        index = 0
        #Add the sarcastic tweets & labels to the train set
        for tweet in sarc_tweets_train:
            if(cluster_assignments[index] == i):
                sarc_count+=1
                train_tweets.append(tweet)
                train_labels.append(1)
            index+=1

        #Add the norm tweets & labels to the train set
        norm_total = 1.3 * sarc_count	#This might be a value that you want to experiment with
        norm_count = 0
        for tweet in norm_tweets_train:
            if(norm_count > norm_total):
                break
            if(random.random() > probability):
                train_tweets.append(tweet)
                train_labels.append(0)
                norm_count+=1

        max_features = 1000

        #Create the MultinomialNB Classifier which you'll be using
        t = True
        if(t):
            print("Model: " + str(i) + ", Training Size: " + str(sarc_count+norm_count))
            count_vect = CountVectorizer(decode_error = 'ignore', lowercase = True, ngram_range=(2,3), max_features = max_features, stop_words='english')
            tfidf = TfidfTransformer()
            train_count_vect = count_vect.fit_transform(train_tweets)
            train_tfidf = tfidf.fit_transform(train_count_vect)				
            mNB = MultinomialNB().fit(train_tfidf, train_labels)
            models.append([mNB,count_vect,tfidf])
            #prediction_tfidf(count_vect, tfidf, mNB, test_tweets, test_labels)
        else:
            train_count_vect = count_vect.fit_transform(train_tweets)
            mNB = MultinomialNB().fit(train_count_vect, train_labels)
            prediction_cv(count_vect, mNB, test_tweets, test_labels)
            models.append(mNB)

    predicted = []
    for i in test_tweets:
        train_pred = count_vect.transform(i)
        train_predi = tfidf.transform(train_pred)
        cluster_predict = k_fit.predict(tfidf_vect.transform(i))[0]
        #print(str(cluster_predict))
        model = models[cluster_predict-1]
        cv = model[1].transform([i])
        tf = model[2].transform(cv)
        sarcasm_prediction = model[0].predict(tf)
        predicted.append(sarcasm_prediction[0])

    print(metrics.classification_report(test_labels, predicted, target_names = [str(0),str(1)]))
    print(metrics.confusion_matrix(test_labels, predicted))
    print("\n")
    print( np.mean(predicted == test_labels))

test_set = make_Test_Set('sarc_test.txt','norm_test.txt')
k_models_classifier('sarc_train.txt', 'norm_train.txt', 500, test_set[0], test_set[1])