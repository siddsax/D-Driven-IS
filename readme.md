# Data Driven Instance Selection

### Problem
The problem is to sub-sample datapoints from a large amount of data so that further machine learning models can be efficiently trained on it on large volumes of data.

### Dependencies

* PySpark 
* nltk stopwords 
* sklearn
* TensorFlow


### Steps
First, the data needs to be pre-processed by tokenization and balancing word-distribution.

```bash
python count.py data.txt #Tokenization
python tail.py data.txt {number of final samples} #subsample data to flatten word-distribution
```
This leads to data with following word-distribution.

<img src="https://github.com/siddsax/D-Driven-IS/blob/master/sorted_freq_full.png" width="400"> <img src="https://github.com/siddsax/D-Driven-IS/blob/master/sorted_freq_sloped.png" width="400">

*Left: Original after tokenization. Right: After word-distribution flattening*

After this, a word2vec model needs to be trained over the descriptions to generate good features for models to be used on top of this. We use sentence vectors to represent a description that is the mean of the word2vec features of all the words in the description.

```bash
python3 word2vec_CPU.py data.txt #saves the trained word encodings as word2vec_dict.npy and outputs desc_embeddings.npz as word2vec embeddings of descriptions
```
 
Now a large-scale K-means|| model can be run on this data that also paralyzes the training process. Here the number of clusters and max iterations are two hyper-parameters that need to be set. The option --executorer-memory also needs to be set if working on a cluster with several slaves. 

```bash
spark-submit --driver-memory {num}g np2HDFS.py desc_embeddings.npz
```

**TIP**: In case the slaves go out of memory increase the number of partitions being done. Ideally, when there is ample memory, the ideal number of partitions is twice the number of cores in the machine. To check if paralyzation is working, it is a good idea to check the utilization of all the CPU cores.

```bash
spark-submit --driver-memory {num}g kmeans-save.py HDFSdesc_embeddings.npz {Number of clusters} { Max no of iterations} 
```
This previous code, create a folder *Final_Output_CLusters* where the model is saved. This can now be used to classify new the data points to different clusters. We do this for the descriptions stored in the file *HDFSdesc_embeddings.npz* 

```bash
spark-submit sampling-kmeans.py $DiscriptionFile Final_Output_Clusters
```

This leads to clusters with word distributions like as the following. Here one cluster has repeated occurrence of words that make sense together, which is learned from the data

<img src="https://github.com/siddsax/D-Driven-IS/blob/master/c3.png" width="400"> <img src="https://github.com/siddsax/D-Driven-IS/blob/master/c5.png" width="400">


#### Supplementary Code
Some supplementary code of additional processing and plotting.

To plot word-frequency graphs for clusters as above for clusters. Run the following scripts, after setting variable array in line #20 as the list of clusters to be visualized.

```bash
python graph_script.py descriptions.txt 
```
It outputs cluster_{number}_word_dist.txt

```bash
python grapher.py cluster_{number}_word_dist.txt
```

The initial attempt to use the Gaussian Mixture Model is also present in the folder GMM. It can be useful to produce better clusters on smaller data.
```bash
python GMM.py {data.csv} {n_cluster} {total sampled points required}    
```

Further to analyze the output of word2vec and clustering K-Nearest Neighbour comes in handy. 
To use it the following two modes are there.
Mode 'a' :- Find the neighbour number of a data point wrt another data point
```bash
python np2csv.py encodings.npz
python KNN.py encodings.npz.csv {number of neighbours} {1st data point} 'a' {2nd Data Point}
```

Mode 'b' is to run see the different neighbours of the 1st data point 
```bash
python KNN.py encodings.npz.csv {number of neighbours} {1st data point} 'b' {Descriptions_file.txt}
```
