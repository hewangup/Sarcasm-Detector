import re

PATH_TO_DATA = '/Users/hewang/Desktop/sarcasm_tweets_text.txt'

fh = open(PATH_TO_DATA)
lines = fh.readlines()
fh.close()

f = open('preprocess.txt','w')

for line in lines:
	s1 = re.sub(r'#\w+ ?', '', line)
	s2 = re.sub(r'@\w+ ?', '', s1)
	s3 = re.sub(r'http\S+', '', s2)
	s4 =re.sub(r'u\w+ ?', '', s3)
	s5 = s4.strip('\n')
	s6 = s5.replace('\n', '')
	f.write(s6)

f.close()
