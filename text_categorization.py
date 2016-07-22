from gensim import corpora, models, similarities
import logging, gensim, bz2, json, re
from nltk.corpus import stopwords
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#Name of the input file
storage_file = 'compiled_tweets_sarcasm.txt'


documents = []
with open(storage_file) as readMe:
    for line in readMe:
        tweet = str(json.loads(line))
        documents.append(tweet)

stoplist = set("for a of the and to in is so it on my with be we he she if they you how at this what are i are it's i'm he's she's that have will all me when get was who what when where why can as an its that's those too than not or do his has like no can't you're oh don't i'll well didn't their from by now some had one were yeah did might still ha really wasn't dont that that's nothing good great doing getting says well hate it help real thing things doesn't even best going been only totally want makes probably there more".split())
#stoplist = set("for a of the end".split())
#stoplist = stopwords.words('english')
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

#Takes Tokens that (have frequency > 1) && !(Stop Words) 
texts = [[token for token in text if frequency[token] > 1] for text in texts]

#Saves dict file
dictionary = corpora.Dictionary(texts)
dictionary.save('tweet_compilation.dict')

#Saves corpus iterator
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('tweet_compilation.mm', corpus) #Stores to disk

#corpus = corpora.MmCorpus('tweet_compilation.mm')
#Loads corpus from memory

#Change this to change the number of topics you want the algorithm to find
num_topics = 10


topics_1 = 'sample_1.txt'
topics_2 = 'sample_2.txt'

tfidf = gensim.models.TfidfModel(corpus)
corpus = tfidf[corpus]

run = True
if(run):
    writeMe = open(topics_1, 'w')
    #lsi = gensim.models.lsimodel.LsiModel(corpus = gensim.models.TfidfModel(corpus), id2word = dictionary, num_topics = num_topics)	
    lsi = gensim.models.lsimodel.LsiModel(corpus = corpus, id2word = dictionary, num_topics = num_topics)
    for i in range (0, num_topics):
        s = ''
        list = lsi.show_topic(i, topn=10)
        for l in list:
            s+=(l[0])
            s+=" "
    #    writeMe.write(str(lsi.show_topic(i, topn=25))+'\n')
        writeMe.write(json.JSONEncoder().encode(s)+'\n')


run = True
if(run):
    writeMe = open(topics_2, 'w')
    lda = gensim.models.LdaModel(corpus = corpus, num_topics = num_topics, id2word = dictionary)
    #lda = gensim.models.LdaModel(corpus = gensim.models.TfidfModel(corpus), num_topics = num_topics, id2word = dictionary)
    for i in range (0, num_topics):
        s = ''
        list = str(lda.print_topic(i, topn=10))
        #writeMe.write(list)
        #writeMe.write('\n')
        list = re.findall('[abcdefghijklmnopqrstuvwxyz]+\'?[abcdefghijklmnopqrstuvwxyz]+', list)
        size = len(list)
        for l in list:	
            s+=l
            s+=" "
        #writeMe.write(str(list)+" "+str(size)+'\n')
        writeMe.write(s+'\n')