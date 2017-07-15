# Instance Selection on Big Data

[Link to report](http://ec2-35-161-213-44.us-west-2.compute.amazonaws.com/EnvestnetYodlee/TDE/blob/samping_siddsax/report.pdf)

Pipeline for running the code accompanying our work in instance selection.

### Problem

Extract descriptions from the whole dataspace that form archetypes in a data-driven manner.

Tasks done by this code
 * Extract descroptions from the data
 * Do some basic tokenization like removing special characters and converting to small letters. More importantly it removes the duplicates after this.
 * Get vectors from the descriptions using word2vec
 * Use those vectors to cluster the data using K-means clustering implemented in spark that is as efficient as possible.
 * Cluster new data using the model learned


And here's the code! :+1:

```bash
cut -d’|’ -f{column number} {Data file} > data.txt
```
It extracts the required column from the data
```bash
python count.py data.txt
```
count.py does the basic tokenization as mentioned above
```bash
python tail.py data.txt {number of final samples}
```
tail.py further samples the data from the input file so that the output contains a flatter word distribution rather than the exponential distribution in the original data.

```bash
python3 word2vec_CPU.py data.txt #saves the trained word encodings as word2vec_dict.npy and outputs desc_embeddings.npz as embeddings
```

In case any error due to encodings while reading a file in python use 
```bash
iconv -f us-ascii -t utf-8 Input file -c tb > Output File
```
It converts the input to utf-8 and removing characters that are in some other encoding.


It learns vectors for the input descriptions and then outputs their embeddings and the learned dictionary. The size of the dictionary is by default the number of words in data although it can be set as any value as the second argument, although we highly recommend to use the dictionary as the all the words in data.

```bash
python word2vec_vectorize.py {data} {saved model/dictionary as .npz}
```
This is a utility tool to convert get vectors using a previosuly trained model as done in the last step.

An important point in regard to word2vec is that some descriptions have no words left afte the tokenization done in it, hence it leads to NANs which are automatically removed from the output but the inital description still has those descriptions. To identify them, word2vec code also outputs a NAN.txt file which has line numbers that give NAN as output, after removing these lines the text data can also be used for analysis.


The next part needs to be done on an EMR cluster. A note before running the code is to edit the following file and restart the resource as shown below.

```bash
sudo vim /etc/hadoop/conf/capacity-scheduler.xml
# Replace line 36 : <value>org.apache.hadoop.yarn.util.resource.DefaultResourceCalculator</value> with <value>org.apache.hadoop.yarn.util.resource.DominantResourceCalculator</value>
sudo /sbin/stop hadoop-yarn-resourcemanager
sudo /sbin/start hadoop-yarn-resourcemanager
```

This converts the data to HDFS so that it can be loaded in spark without occupying large amount of RAM later. The following step needs to be done in a cluster with large RAM and set most of it as driver memory.  
```bash
spark-submit --driver-memory {num}g np2HDFS.py desc_embeddings.npz
```

The following code trains the K-means model on the data and then outputs it. The hyperparameters are the number of clusters and max iterations. For 15 Million datasize we worked with 10K clusters and 100 iterations. Each iteration takes 7 minutes with an initialization step also taking a lot of time as it is done using K-means|| or scalable K-means++.

The option --executorer-memory also needs to be set if working on a cluster with several slaves. In case the slaves go out of memory increase the number of partitions being done. Ideally when there is ample meory, the ideal number of partions is twice the number of cores in the machine. To check if parallyzation is working, it is a good idea to check the utilization of all the cpu cores.

```bash
spark-submit --driver-memory {num}g kmeans-save.py HDFSdesc_embeddings.npz {Number of clusters} { Max no of iterations} 
```

Finally we need to classify the data points in the cluters using the petrained model.
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
