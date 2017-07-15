# USAGE : python3 word2vec_PICKLE desc.zip
# Output: word2vec_embeddings.pickle
#matplotlib inline

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

file = codecs.open(sys.argv[1],"r",encoding='utf-8')


#Build the dictionary and replace rare words with UNK token.

#if(len(sys.argv)==3):
#  vocabulary_size = int(sys.argv[2])
stop = set(stopwords.words('english'))
tknzr = TweetTokenizer()


def build_dataset():
  count = [['UNK', -1]]
  number_of_samples=0
  allWords=[]
  for line in file:
    number_of_samples=number_of_samples+1
# lowercase the sentence and tokenize it
    word=tknzr.tokenize(line.lower())
# remove special symbols
   # words=[i for i in word if i not in symbols] 
#Remove single-character tokens (mostly punctuation)
    words = [i for i in word if len(i) > 1] 
# Remove numbers
    words = [i for i in words if not i.isnumeric()]

# stemming .. but it may make words worst so commenting
#    stemmer = SnowballStemmer("english")
#    words = [stemmer.stem(i) for i in word]
 
# Remove stopwords
    words_after_stopwords=[i for i in words if i not in stop]  # remove stopwords
    allWords.extend(words_after_stopwords)
  allWordsDist = nltk.FreqDist(w.lower() for w in allWords)
  print('Data size %d' % len(allWords))
  cnt = collections.Counter(allWords)
  vocabulary_size = len(cnt)
  print("num words=" + str(vocabulary_size))
  f = open("num words","w")
  f.write(str(vocabulary_size))
  if(len(sys.argv)==3):
    vocabulary_size = int(sys.argv[2])
  count.extend(cnt.most_common(vocabulary_size - 1))

  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  return data, count, dictionary, reverse_dictionary,allWords,number_of_samples,vocabulary_size

data, count, dictionary, reverse_dictionary,allWords,number_of_samples,vocabulary_size = build_dataset()
if(len(sys.argv)==3):
  vocabulary_size = int(sys.argv[2])
print(len(data))
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
print('number of samples',number_of_samples)

#Function to generate a training batch for the skip-gram model.

data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

#print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

#Train a skip-gram model.

batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. 
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.

graph = tf.Graph()

with tf.device('/cpu:0'):

  # Input data.
  train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  
  # Variables.
  embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  softmax_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                         stddev=1.0 / math.sqrt(embedding_size)))
  softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Model.
  # Look up embeddings for inputs.
  embed = tf.nn.embedding_lookup(embeddings, train_dataset)
  # Compute the softmax loss, using a sample of the negative labels each time.
  loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                               labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

  # Optimizer.
  # Note: The optimizer will optimize the softmax_weights AND the embeddings.
  # This is because the embeddings are defined as a variable quantity and the
  # optimizer's `minimize` method will by default modify all variable quantities 
  # that contribute to the tensor it is passed.
  # See docs on `tf.train.Optimizer.minimize()` for more details.
  optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
  
  # Compute the similarity between minibatch examples and all embeddings.
  # We use the cosine distance:
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
NAN_counter = 0
num_steps = 10000
config = tf.ConfigProto(allow_soft_placement=True)
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
#sess = tf.Session(config=config)
#f1=open('NANs', 'w+')

with tf.Session(config=config) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  average_loss = 0
  for step in range(num_steps):
    batch_data, batch_labels = generate_batch(
      batch_size, num_skips, skip_window)
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += l
    if step % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step %d: %f' % (step, average_loss))
      average_loss = 0
    # note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log = '%s %s,' % (log, close_word)
        print(log)
  final_embeddings = normalized_embeddings.eval()

####CREATE VECTOR FOR DESCRIPTION USING WORD VECTORS
  print("process Desc2vec started....")
  # create word2vec dictionary for words and their vectors
  print("creating word2vec dict")
  word_2_vec={}
  i =0
  f1=open('NANs.txt', 'w+')
  for word,_ in collections.Counter(allWords).most_common(vocabulary_size - 1):
    word_2_vec[word]=final_embeddings[i]
    i=i+1
    #print(i)
    #f1.write(word)
  #sys.exit()
  # array for description embeddings
  desc_embeddings=np.zeros([number_of_samples,embedding_size])
  #file_2 = codecs.open(sys.argv[1],"r",encoding="utf-8")
  sample=0
  z = 0
  #with open("word2vec_dict", 'w') as f:
   # json.dump(word_2_vec, f)
  np.save('word2vec_dict', word_2_vec)
  file = codecs.open(sys.argv[1],"r",encoding='utf-8') 
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
      NAN_counter = NAN_counter +1
      continue
    #if sample==0:
     # desc_embeddings=desc_array
    #else:
     # desc_embeddings=np.concatenate((desc_embeddings,desc_array),axis=0)
    desc_embeddings[sample,:] = desc_array

    sample=sample+1
  n = number_of_samples - NAN_counter
  desc_embeddings = desc_embeddings[0:n,:]
  #DESCRIPTION EMBEDDINGSi
  print(vocabulary_size)
  with open('desc_embeddings.npz','wb') as f:
    np.save(f,desc_embeddings)
  #WORD2VEC EMBEDDINGS
  with open('word2vec_embeddings.pickle','wb') as f:
    pickle.dump(final_embeddings,f,protocol=4)
