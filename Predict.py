import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import re
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import gensim
from gensim.models import Word2Vec 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
#from keras.layers import Input
from keras.layers.convolutional import Conv1D
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
import nltk
import tweepy
from textblob import TextBlob
from keras.models import load_model
import sys
#import tokenizer

#nltk.download('punkt')
MAX_SEQUENCE_LENGTH=50
EMBEDDING_DIM=300

consumer_key=''
consumer_secret=''

access_token=''
access_secret=''
# Authenticating Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)



def remove_punct(text):
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', '', text)
    return text_nopunct

def lower_token(tokens): 
    return [w.lower() for w in tokens]   

def removeStopWords(tokens): 
    return [word for word in tokens if word not in stoplist]

#Collecting 200 tweets from titter on Specific topic	
api= tweepy.API(auth)
public_tweets = api.user_timeline(screen_name=str(sys.argv[1]),
                           # 200 is the maximum allowed count
                           count=200,
                           include_rts = False,
                           # Necessary to keep full_text 
                           # otherwise only the first 140 words are extracted
                           tweet_mode = 'extended'
                           )
data={'tweet':[],'Polarity':[]}

#Storing it in DataFrame
dat1 = pd.DataFrame(data)

#Calculating the Polarity using TextBlob
for tweet in public_tweets:
    #print(tweet.full_text)
    analysis=TextBlob(tweet.full_text)
    len(tweet.full_text)
    if analysis.sentiment.polarity<0:
        polar=0
    else:
        polar=4
    t={'tweet':tweet.full_text,'Polarity':polar}
   
    dat1=dat1.append(t,ignore_index=True)
   
    
#Removing Punctuations and other Extra Characters from the Text
dat1['Text_Clean'] = dat1['tweet'].apply(lambda x: remove_punct(x))
tokens = [word_tokenize(sen) for sen in dat1.Text_Clean]
lower_tokens = [lower_token(token) for token in tokens]

stoplist = stopwords.words('english')
filtered_words = [removeStopWords(sen) for sen in lower_tokens]
dat1['Text_Final']=[' '.join(sen) for sen in filtered_words]
dat1['tokens'] = filtered_words    


#Performing One-Hot Encoding to Feed to the Network
pos = []
neg = []
for l in dat1.Polarity:
    if l == 0:
        pos.append(0)
        neg.append(1)
    elif l == 4:
        pos.append(1)
        neg.append(0)
        
dat1['Pos']= pos
dat1['Neg']= neg

dat1 = dat1[['Text_Final', 'tokens', 'Polarity', 'Pos', 'Neg']]


all_test_words = [word for tokens in dat1["tokens"] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in dat1["tokens"]]
TEST_VOCAB = sorted(list(set(all_test_words)))


tokenizer = Tokenizer(num_words=len(TEST_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(dat1["Text_Final"].tolist())

test_sequences = tokenizer.texts_to_sequences(dat1["Text_Final"].tolist())
test_dat = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

#print(test_dat)
num_epochs = 50
batch_size = 32

#Loading the Prvious Trained Model
model=load_model('C:/wamp64/www/Project/MyTwitModel.h5')

predictions = model.predict(test_dat, 
                            batch_size=1024, 
                            verbose=1)
labels = [4, 0]
prediction_labels=[]

#Appending the Predicted Labels
for p in predictions:
    #print(p)
    prediction_labels.append(labels[np.argmax(p)])
sum(dat1.Polarity==prediction_labels)/len(prediction_labels)

#Calculating the Percentage of Positive tewwtes and negative tweets
pos=0;
neg=0;
for p in prediction_labels:
    if p==4:
        pos=pos+1
    else:
        neg=neg+1
PosPer=float(pos/len(prediction_labels))
NegPer=neg/len(prediction_labels)

print(PosPer)
print(NegPer)