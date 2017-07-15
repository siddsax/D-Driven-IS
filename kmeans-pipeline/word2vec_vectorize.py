from __future__ import print_function
import collections
import json
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
import pickle
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
import sys
from nltk.tokenize import TweetTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import codecs
stop = set(stopwords.words('english'))
tknzr = TweetTokenizer()
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.

with tf.Session(config=config) as session:
  i =0
  f1=open('NANs.txt', 'w+')
  word_2_vec = np.load(sys.argv[2]).item()
  file = codecs.open(sys.argv[1],"r",encoding='utf-8') 
  z = 0
  sample=0
  number_of_samples = 0
  for line in file:
    number_of_samples=number_of_samples+1
  desc_embeddings=np.zeros([number_of_samples,embedding_size])
  
  file = codecs.open(sys.argv[1],"r",encoding='utf-8')
  NAN_counter = 0
  for line in file:
    if(z%1000==0):
      print(z)
    z = z + 1
    word=tknzr.tokenize(line.lower())
    words = [i for i in word if len(i) > 1] 
    words = [i for i in words if not i.isnumeric()]
# stemming .. but it may make words worst so commenting
#    stemmer = SnowballStemmer("english")
#    words = [stemmer.stem(i) for i in word]
    words_after_stopwords=[i for i in words if i not in stop]  # remove stopwords
    desc=np.zeros([embedding_size])
    unkmember=np.zeros([embedding_size])
    num_of_words=0
    # add word vectors of all words in description and take the average
    #NOTE-------------------------------------------- another way is weighted average using TFIDF (NEED TO IMPLEMENT)
    #print(words_after_stopwords)
    for word in words_after_stopwords:
      vec=word_2_vec.get(word,unkmember)
      desc=vec+desc
      num_of_words=num_of_words+1
    #print(num_of_words) # CAUSING FORMATION OF NAN
    desc=desc/num_of_words
    # Convert to array..needed for concat operation
    desc_array=np.array([desc])
    if(np.isnan(desc_array).any()):
      f1.write(str(z)+'\n')
      print("ASSDADASDSADSADSDSADSADD")
      continue
      NAN_counter = NAN_counter + 1
    desc_embeddings[sample,:] = desc_array

    sample=sample+1
  n = number_of_samples - NAN_counter
  desc_embeddings = desc_embeddings[0:n,:]
  #DESCRIPTION EMBEDDINGS
  #print(vocabulary_size)
  with open('desc_embeddings.npz','wb') as f:
    np.save(f,desc_embeddings)
  #WORD2VEC EMBEDDINGS
  #with open('word2vec_embeddings.pickle','wb') as f:
   # pickle.dump(final_embeddings,f,protocol=4)
