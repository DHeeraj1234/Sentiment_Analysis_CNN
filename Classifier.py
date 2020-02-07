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

MAX_SEQUENCE_LENGTH=50
EMBEDDING_DIM=300



def remove_punct(text):
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', '', text)
    return text_nopunct

def lower_token(tokens): 
    return [w.lower() for w in tokens]   

def removeStopWords(tokens): 
    return [word for word in tokens if word not in stoplist]
	
#Loading the Dataset into DataFrame	
dat = pd.read_csv("Datatw.csv",engine='python')

dat.shape



#Cleaning the Data by Removing Extra Characters
dat['Text_Clean'] = dat['text'].apply(lambda x: remove_punct(x))
tokens = [word_tokenize(sen) for sen in dat.Text_Clean]
lower_tokens = [lower_token(token) for token in tokens]

stoplist = stopwords.words('english')
filtered_words = [removeStopWords(sen) for sen in lower_tokens]
dat['Text_Final'] = [' '.join(sen) for sen in filtered_words]
dat['tokens'] = filtered_words

dat.head()

#Performing OneHot encoding where 0 is negative and 4 is positive
pos = []
neg = []
for l in dat.target:
    if l == 0:
        pos.append(0)
        neg.append(1)
    elif l == 4:
        pos.append(1)
        neg.append(0)
print(len(pos))
print(len(neg))        
dat['Pos']= pos
dat['Neg']= neg

dat = dat[['Text_Final', 'tokens', 'target', 'Pos', 'Neg']]
dat.head()

#Splpitting the Data for Training and Testing
data_train, data_test = train_test_split(dat, 
                                         test_size=0.15, 
                                         random_state=42)	
										 
										 all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in data_train["tokens"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))


all_test_words = [word for tokens in data_test["tokens"] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in data_test["tokens"]]
TEST_VOCAB = sorted(list(set(all_test_words)))


word2vec_path = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(data_train["Text_Final"].tolist())
training_sequences = tokenizer.texts_to_sequences(data_train["Text_Final"].tolist())
train_word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(train_word_index))
train_cnn_data = pad_sequences(training_sequences, 
                               maxlen=MAX_SEQUENCE_LENGTH)


train_embedding_weights = np.zeros((len(train_word_index)+1, 
 EMBEDDING_DIM))
for word,index in train_word_index.items():
    train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
print(train_embedding_weights.shape)

test_sequences = tokenizer.texts_to_sequences(data_test["Text_Final"].tolist())
test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

#Defining the Convolution Neural Network

def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
 
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=False)
    
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    convs = []
    filter_sizes = [2,3,4,5,6]
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=200, 
                        kernel_size=filter_size, 
                        activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)
    l_merge = concatenate(convs, axis=1)
    x = Dropout(0.1)(l_merge)  
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    preds = Dense(labels_index, activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model


label_names = ['Pos', 'Neg']
y_train = data_train[label_names].values

x_train = train_cnn_data
y_tr = y_train

model = ConvNet(train_embedding_weights, 
                MAX_SEQUENCE_LENGTH, 
                len(train_word_index)+1, 
                EMBEDDING_DIM, 
                len(list(label_names)))
				
num_epochs = 50
batch_size = 32

#Training the Network
hist = model.fit(x_train, 
                 y_tr, 
                 epochs=num_epochs, 
                 validation_split=0.1, 
                 shuffle=True, 
                 batch_size=batch_size)

#Saving the Model
model.save('C:/Users/dmaddi/Downloads/MyTwitModel.h5')

