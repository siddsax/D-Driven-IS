import sys
import argparse
#from GMMModel import GMMModel
from pyspark import SparkContext, SparkConf
from pyspark.context import SparkContext

from pyspark.mllib.clustering import KMeans,KMeansModel
from pyspark.mllib.random import RandomRDDs
import shutil
import pickle
# import tensorflow as tf
# $example on$
from numpy import array
import numpy as np
# $example off$
import csv
# $example on$
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel
# $example off$
import scipy.stats
import math
from random import randint
from operator import itemgetter
import os
from numpy import genfromtxt
from pyspark.sql.functions import *
from pyspark.mllib.linalg import Vectors

def parseVector(line):
    return np.array([float(x) for x in line.split(',')])

if __name__ == "__main__":

    b = open("name", 'wb')
    sc = SparkContext("local[*]","kmeans")

    print("data being loaded.....")
    data = sc.textFile(sys.argv[1]).map(lambda row: map(lambda x: float(x), row.split(',')))
#file:///dev/desc_hdfs
    print("data loaded!")
    D = 128
    print("loading and counting")
    data_size = data.count()
    print("count done")
    print("model being loaded.....")
    model =  KMeansModel.load(sc,sys.argv[2])    
    print("model loaded!")

    centers = model.clusterCenters
    # ################SAMPLING##################################################
    #total_sampled_points = int(sys.argv[3])
    cluster = {}
    samples = {}
    print("data being stored in array....")
    #da = data.collect()
    print("data stored")
    
    n_clusters = model.k

    for j in range(n_clusters):
        cluster[j] = []
    print("adding data in clusters")
    data = data.zipWithIndex()
    labels_dist =np.asarray(data.map(lambda (x,i): (model.predict(x),np.linalg.norm(centers[model.predict(x)]-(x)),i)).collect())
    #label = labels_dist.map(lambda x : x[0]).collect()
    #label_dist = label_dist[label_dist[:,1].argsort()]
    #print(labels_dist[0:2]) 
    for x in range(data_size):
        cluster[labels_dist[x,0]].append(x)
    print("dividing to clusters done")
    # Sort the clusters by their soft probabilities (NOT USED NOW)
    print("Sorting Now")
    for j in range(n_clusters):
       print(j)
       a = labels_dist[cluster[j],:]
       a = np.asarray(a[a[:,1].argsort()])
       cluster[j] = a[:,2]#labels_dist.filter(lambda (cluster,dist,i) : cluster == j).sortBy(lambda x: x[1]).map(lambda (x,k,i) : i).collect()
       #data.zipWithIndex().filter(lambda (key,index) : index == cluster[j]).map(lambda (x,k) : (np.linalg.norm(centers[j]-(x)),k)).sortBy(lambda x: x[0]).map(lambda (x,k) : k).collect()
       #values = tuple(1.0/(np.linalg.norm(centers[j]-(data.zipWithIndex().filter(lambda y : x == y[1]).map(lambda g : g[0]).first()))+.1) for x in cluster[j])
       #values = tuple(np.linalg.norm(centers[j]-(da[x])) for x in cluster[j])
        
       # values = probabs[]#corresponding values
       # cluster[j] = [x for (y, x) in sorted(zip(values, cluster[j]))]
    print("dividing sorting done")
    # print("Sampling")
    # for i in range(n_clusters):
    #     samples[i] = {}
    #     size = int(len(cluster[i]))
    #     fraction = float(size) / data_size
    #     #print(total_sampled_points)
    #     num_sampled_points = int(math.floor(total_sampled_points * fraction))
    #     if (size == 0):
    #         continue

    #     indices = np.random.randint(size, size=num_sampled_points)
    #     if(num_sampled_points == 0):
    #         continue
    #     data_indices = itemgetter(*indices)(cluster[i]) 
    #     if isinstance(data_indices, int):
    #         data_indices = [data_indices]
    #     print(size)
    #     #for ind in data_indices:
    #     samples[i] = data.zipWithIndex().filter(lambda (key,index) : index == data_indices).map(lambda (k,i) : k).collect()

    # # save samples
    # print("Sampled")
    
    # name = 'samples.csv_' + str(n_clusters)
    # print("writing samples points")
    # with open(name, 'wb') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for key, value in samples.items():
    #        writer.writerow([key, value])
    # print("wrote samples")
    
    print("cluster points writing")
    name = 'cluster_points_sorted.csv_' + str(n_clusters)
    with open(name, 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in cluster.items():
               writer.writerow([key, value])

    print("wrote all points to file")

    sc.stop()



