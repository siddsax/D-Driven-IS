# Data Driven Instance Selection

### Problem

Extract descriptions from the data and create a summary of the data.

Tasks done by this code
 * Extract descroptions from the data
 * Do some basic tokenization like removing special characters and converting to small letters. More importantly it removes the duplicates after this.
 * Get vectors from the descriptions using word2vec
 * Use those vectors to cluster the data using K-means clustering implemented in spark that is as efficient as possible.
 * Cluster new data using the model learned

Steps to run the code.

```bash
python count.py data.txt #Tokenization
python tail.py data.txt {number of final samples} #subsample data to flatten word-distribution
python3 word2vec_CPU.py data.txt #saves the trained word encodings as word2vec_dict.npy and outputs desc_embeddings.npz as word2vec embeddings of descriptions
```
In case any error due to encodings while reading a file in python use 
```bash
iconv -f us-ascii -t utf-8 Input file -c tb > Output File #converts the input to utf-8
```

```bash
spark-submit --driver-memory {num}g np2HDFS.py desc_embeddings.npz
```

The following code trains the K-means model on the data and then outputs it. The hyperparameters are the number of clusters and max iterations. It uses K-means|| aka scalable K-means++.

The option --executorer-memory also needs to be set if working on a cluster with several slaves. In case the slaves go out of memory increase the number of partitions being done. Ideally when there is ample meory, the ideal number of partions is twice the number of cores in the machine. To check if parallyzation is working, it is a good idea to check the utilization of all the cpu cores.

```bash
spark-submit --driver-memory {num}g kmeans-save.py HDFSdesc_embeddings.npz {Number of clusters} { Max no of iterations} 
```

Classify the data points in the cluters using the petrained model.
```bash
spark-submit sampling-kmeans.py HDFSdesc_embeddings.npz Final_Output_Clusters/
```

#### Supplementary Code

Add the index numbers of the clusters in line 20 and then the code finds the word frequency of the data points within it, then it can be fed to grapher.py to plot it.
```bash
python graph_script.py descriptions.txt 
```
It outputs cluster_{number}_word_dist.txt

```bash
python grapher.py cluster_{number}_word_dist.txt
```

The intial attempt to use Gaussian Mixture Model is also present in the folder GMM. It can be useful to produce better clusters on smaller data.
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
